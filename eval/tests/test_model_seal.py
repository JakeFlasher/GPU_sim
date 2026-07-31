from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from eval.contracts._common import (
    AuthorityBlockerCode,
    EvaluatorAuthorityUnavailable,
)
from eval.contracts.model_seal import (
    MODEL_SEAL_SCHEMA_ID,
    MODEL_SEAL_SCHEMA_SHA256,
    RetainedPayloadSnapshot,
    canonical_model_bytes,
    create_detached_model_seal,
    load_detached_model_seal,
    require_nfc,
    require_unique_identities,
    seal_evaluator_model,
    strict_json_object,
    verify_detached_model_seal,
    verify_evaluator_model,
    verify_retained_schema,
)
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    read_regular_snapshot,
    seal_model,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "eval" / "schemas" / "model_seal.v1.schema.json"
_PINNED_SCHEMA = read_regular_snapshot(SCHEMA_PATH)
SCHEMA_BYTES, SCHEMA_SNAPSHOT = RetainedPayloadSnapshot.read_path(SCHEMA_PATH)


def _sealed_model(payload: object | None = None) -> dict[str, object]:
    return seal_evaluator_model(
        {
            "schema_version": 1,
            "artifact_kind": "TEST_EVALUATOR_MODEL",
            "schema_sha256": MODEL_SEAL_SCHEMA_SHA256,
            "payload": {"z": "last", "a": [1, 2, 3]} if payload is None else payload,
        }
    )


def _schema_arguments() -> dict[str, object]:
    return {
        "schema_bytes": SCHEMA_BYTES,
        "schema_snapshot": SCHEMA_SNAPSHOT,
    }


def _detached_arguments() -> dict[str, object]:
    return {
        **_schema_arguments(),
        "sealed_schema_bytes": SCHEMA_BYTES,
        "sealed_schema_snapshot": SCHEMA_SNAPSHOT,
        "sealed_schema_id": MODEL_SEAL_SCHEMA_ID,
    }


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_BYTES)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_schema_bytes_are_pinned_to_runtime_digest_and_identity() -> None:
    assert sha256_bytes(SCHEMA_BYTES) == MODEL_SEAL_SCHEMA_SHA256
    schema = verify_retained_schema(
        SCHEMA_BYTES,
        SCHEMA_SNAPSHOT,
        expected_sha256=MODEL_SEAL_SCHEMA_SHA256,
        expected_id=MODEL_SEAL_SCHEMA_ID,
    )
    assert schema["$id"] == MODEL_SEAL_SCHEMA_ID

    with pytest.raises(ImmutableArtifactError, match="does not match snapshot"):
        verify_retained_schema(
            SCHEMA_BYTES + b"\n",
            SCHEMA_SNAPSHOT,
            expected_sha256=MODEL_SEAL_SCHEMA_SHA256,
            expected_id=MODEL_SEAL_SCHEMA_ID,
        )


def test_canonical_model_seal_is_deterministic_across_key_order() -> None:
    first = seal_evaluator_model(
        {
            "schema_version": 1,
            "artifact_kind": "TEST_EVALUATOR_MODEL",
            "schema_sha256": MODEL_SEAL_SCHEMA_SHA256,
            "payload": {"z": 3, "a": 1},
        }
    )
    second = seal_evaluator_model(
        {
            "payload": {"a": 1, "z": 3},
            "schema_sha256": MODEL_SEAL_SCHEMA_SHA256,
            "artifact_kind": "TEST_EVALUATOR_MODEL",
            "schema_version": 1,
        }
    )

    assert first == second
    assert canonical_model_bytes(first) == canonical_model_bytes(second)
    verify_evaluator_model(
        first,
        expected_schema_version=1,
        expected_artifact_kind="TEST_EVALUATOR_MODEL",
        expected_schema_sha256=MODEL_SEAL_SCHEMA_SHA256,
        expected_schema_id=MODEL_SEAL_SCHEMA_ID,
        **_schema_arguments(),
    )


def test_strict_json_rejects_duplicates_nonfinite_unsafe_and_surrogates() -> None:
    for payload, message in (
        (b'{"schema_version":1,"schema_version":1}', "duplicate JSON key"),
        (b'{"outer":{"identity":"a","identity":"b"}}', "duplicate JSON key"),
        (b'{"value":NaN}', "non-finite"),
        (b'{"value":9007199254740992}', "safe range"),
        (b'{"value":"\\ud800"}', "unpaired surrogate"),
    ):
        with pytest.raises(ImmutableArtifactError, match=message):
            strict_json_object(payload)


def test_nfc_and_control_characters_are_rejected() -> None:
    assert require_nfc("é", "name") == "é"
    with pytest.raises(ImmutableArtifactError, match="NFC-normalized"):
        require_nfc("é", "name")
    with pytest.raises(ImmutableArtifactError, match="forbidden control"):
        require_nfc("name\n", "name")


def test_retained_snapshot_binds_exact_payload_and_metadata() -> None:
    SCHEMA_SNAPSHOT.verify_payload(SCHEMA_BYTES)
    with pytest.raises(EvaluatorAuthorityUnavailable) as serialized_production:
        RetainedPayloadSnapshot.from_model(SCHEMA_SNAPSHOT.to_model())
    assert (
        serialized_production.value.code
        is AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED
    )
    fixture = RetainedPayloadSnapshot.from_fixture_payload(
        SCHEMA_BYTES, _PINNED_SCHEMA.metadata
    )
    assert RetainedPayloadSnapshot.from_model(fixture.to_model()) == fixture

    with pytest.raises(ImmutableArtifactError, match="does not match snapshot"):
        SCHEMA_SNAPSHOT.verify_payload(SCHEMA_BYTES + b" ")

    changed = fixture.to_model()
    changed["metadata"]["inode"] = str(int(changed["metadata"]["inode"]) + 1)
    with pytest.raises(ImmutableArtifactError, match="identity SHA-256 mismatch"):
        RetainedPayloadSnapshot.from_model(changed)


def test_read_path_records_real_descriptor_acquisition_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    retained_path = tmp_path / "retained.json"
    retained_path.write_bytes(SCHEMA_BYTES)
    payload, snapshot = RetainedPayloadSnapshot.read_path(retained_path)
    assert payload == SCHEMA_BYTES
    assert snapshot.is_production_acquisition
    assert snapshot.to_model()["acquisition"] == "PINNED_REGULAR_NOFOLLOW_V1"
    snapshot.require_production_acquisition("test retained payload")
    snapshot.verify_payload(payload)

    alias = tmp_path / "retained-alias.json"
    alias.symlink_to(retained_path)
    with pytest.raises(ImmutableArtifactError):
        RetainedPayloadSnapshot.read_path(alias)


def test_bytes_plus_metadata_constructor_cannot_claim_descriptor_acquisition() -> None:
    with pytest.raises(EvaluatorAuthorityUnavailable) as transferred:
        RetainedPayloadSnapshot.from_pinned_regular(_PINNED_SCHEMA)
    assert (
        transferred.value.code
        is AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED
    )

    with pytest.raises(EvaluatorAuthorityUnavailable) as direct:
        RetainedPayloadSnapshot(
            acquisition="PINNED_REGULAR_NOFOLLOW_V1",
            payload_sha256=SCHEMA_SNAPSHOT.payload_sha256,
            size_bytes=SCHEMA_SNAPSHOT.size_bytes,
            metadata=SCHEMA_SNAPSHOT.metadata,
            snapshot_identity_sha256=SCHEMA_SNAPSHOT.snapshot_identity_sha256,
        )
    assert (
        direct.value.code
        is AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED
    )

    metadata = _PINNED_SCHEMA.metadata
    with pytest.raises(EvaluatorAuthorityUnavailable) as captured:
        RetainedPayloadSnapshot.from_payload(SCHEMA_BYTES, metadata)
    assert (
        captured.value.code
        is AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED
    )

    fixture = RetainedPayloadSnapshot.from_fixture_payload(SCHEMA_BYTES, metadata)
    assert fixture.to_model()["acquisition"] == "FIXTURE_CALLER_BYTES_V1"
    with pytest.raises(
        EvaluatorAuthorityUnavailable,
        match="RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED",
    ):
        fixture.require_production_acquisition("hostile fixture")


def test_sealing_requires_schema_hash_and_rejects_preexisting_seal() -> None:
    with pytest.raises(ImmutableArtifactError, match="schema_sha256"):
        seal_evaluator_model(
            {"schema_version": 1, "artifact_kind": "TEST_EVALUATOR_MODEL"}
        )
    with pytest.raises(ImmutableArtifactError, match="unexpectedly contains"):
        seal_evaluator_model(
            {
                "schema_version": 1,
                "artifact_kind": "TEST_EVALUATOR_MODEL",
                "schema_sha256": MODEL_SEAL_SCHEMA_SHA256,
                "model_seal_sha256": "0" * 64,
            }
        )


def test_detached_seal_is_deterministic_and_binds_model_and_schema() -> None:
    model = _sealed_model()
    first = create_detached_model_seal(model, **_detached_arguments())
    second = create_detached_model_seal(copy.deepcopy(model), **_detached_arguments())

    assert first == second
    assert first["schema_sha256"] == MODEL_SEAL_SCHEMA_SHA256
    assert first["sealed_schema_sha256"] == MODEL_SEAL_SCHEMA_SHA256
    verify_detached_model_seal(model, first, **_detached_arguments())
    _validator().validate(first)

    changed = copy.deepcopy(model)
    changed["payload"]["a"].append(4)  # type: ignore[index,union-attr]
    changed.pop("model_seal_sha256")
    changed = seal_model(changed)
    with pytest.raises(ImmutableArtifactError, match="sealed model SHA-256 mismatch"):
        verify_detached_model_seal(changed, first, **_detached_arguments())


def test_detached_seal_rejects_hash_identity_and_schema_byte_replay() -> None:
    model = _sealed_model()
    seal = create_detached_model_seal(model, **_detached_arguments())

    bad_hash = copy.deepcopy(seal)
    bad_hash["sealed_model_sha256"] = "0" * 64
    bad_hash.pop("model_seal_sha256")
    bad_hash = seal_model(bad_hash)
    with pytest.raises(ImmutableArtifactError, match="sealed model SHA-256 mismatch"):
        verify_detached_model_seal(model, bad_hash, **_detached_arguments())

    bad_version = copy.deepcopy(model)
    bad_version["schema_version"] = 2
    bad_version.pop("model_seal_sha256")
    bad_version = seal_model(bad_version)
    with pytest.raises(ImmutableArtifactError, match="schema identity mismatch"):
        verify_detached_model_seal(bad_version, seal, **_detached_arguments())

    replay_arguments = _detached_arguments()
    replay_arguments["sealed_schema_bytes"] = SCHEMA_BYTES + b"\n"
    with pytest.raises(ImmutableArtifactError, match="does not match snapshot"):
        verify_detached_model_seal(model, seal, **replay_arguments)


def test_detached_loader_rejects_duplicates_extra_fields_and_boolean_versions() -> None:
    seal = create_detached_model_seal(_sealed_model(), **_detached_arguments())
    payload = canonical_model_bytes(seal)
    assert load_detached_model_seal(payload, **_schema_arguments()) == seal

    duplicate = payload[:-1] + b',"artifact_kind":"EVALUATOR_MODEL_SEAL"}'
    with pytest.raises(ImmutableArtifactError, match="duplicate JSON key"):
        load_detached_model_seal(duplicate, **_schema_arguments())

    extra = copy.deepcopy(seal)
    extra["extra"] = "forbidden"
    with pytest.raises(ImmutableArtifactError, match="fields are not exact"):
        load_detached_model_seal(json.dumps(extra).encode(), **_schema_arguments())

    boolean = copy.deepcopy(seal)
    boolean["schema_version"] = True
    boolean.pop("model_seal_sha256")
    boolean = seal_model(boolean)
    with pytest.raises(ImmutableArtifactError, match="schema_version"):
        load_detached_model_seal(canonical_model_bytes(boolean), **_schema_arguments())
    with pytest.raises(ValidationError):
        _validator().validate(boolean)

    boolean_inner = copy.deepcopy(seal)
    boolean_inner["sealed_schema_version"] = True
    boolean_inner.pop("model_seal_sha256")
    boolean_inner = seal_model(boolean_inner)
    with pytest.raises(ImmutableArtifactError, match="sealed_schema_version"):
        load_detached_model_seal(
            canonical_model_bytes(boolean_inner), **_schema_arguments()
        )
    with pytest.raises(ValidationError):
        _validator().validate(boolean_inner)


def test_hash_and_identifier_regexes_reject_terminal_newlines() -> None:
    valid = create_detached_model_seal(_sealed_model(), **_detached_arguments())
    validator = _validator()
    validator.validate(valid)

    for field in ("schema_sha256", "sealed_schema_sha256", "sealed_model_sha256"):
        invalid = copy.deepcopy(valid)
        invalid[field] += "\n"
        with pytest.raises(ValidationError):
            validator.validate(invalid)

    invalid_kind = copy.deepcopy(valid)
    invalid_kind["sealed_artifact_kind"] += "\n"
    with pytest.raises(ValidationError):
        validator.validate(invalid_kind)


def test_semantic_identity_collections_reject_duplicates() -> None:
    values = (
        {"namespace": "NCU", "ordinal": 0},
        {"namespace": "NCU", "ordinal": 0},
    )
    with pytest.raises(ImmutableArtifactError, match="duplicate launch identity"):
        require_unique_identities(
            values,
            identity=lambda value: (value["namespace"], value["ordinal"]),
            label="launch",
        )


def test_model_seal_schema_is_closed_and_hash_strict() -> None:
    validator = _validator()
    valid = create_detached_model_seal(_sealed_model(), **_detached_arguments())

    for field, value in (
        ("schema_version", 2),
        ("schema_version", True),
        ("sealed_schema_version", 0),
        ("artifact_kind", "OTHER"),
        ("sealed_model_sha256", "A" * 64),
        ("model_seal_sha256", "0" * 63),
    ):
        invalid = copy.deepcopy(valid)
        invalid[field] = value
        with pytest.raises(ValidationError):
            validator.validate(invalid)

    extra = copy.deepcopy(valid)
    extra["numeric_metric_value"] = 1.0
    with pytest.raises(ValidationError):
        validator.validate(extra)
