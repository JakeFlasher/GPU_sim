from __future__ import annotations

import copy
import json

import pytest

from tools.local_labels import sol_definition_equivalence as equiv


def _bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def _comparison(source: object, staged: object) -> equiv.DefinitionComparison:
    return equiv.compare_definition_bytes(_bytes(source), _bytes(staged))


@pytest.mark.parametrize("field", equiv.OPTIONAL_TOP_LEVEL_FIELDS)
@pytest.mark.parametrize(
    ("source_state", "source_value"),
    [("missing", object()), ("empty_string", ""), ("null", None)],
)
def test_top_level_optional_representation_cases_remain_equivalent(
    field, source_state, source_value
):
    source = {"name": "definition"}
    if source_state != "missing":
        source[field] = source_value
    staged = {"name": "definition", field: None}

    comparison = equiv.require_definition_equivalent(_bytes(source), _bytes(staged))
    state = comparison.attestation["top_level_field_states"][field]

    assert comparison.rule_id == "SOL_DEF_EQUIV_V2"
    assert comparison.difference_pointers == ()
    assert state == {
        "source": source_state,
        "staged": "null",
        "source_normalized": source_state in {"missing", "empty_string"},
        "staged_normalized": False,
    }


@pytest.mark.parametrize(
    ("source_axis", "staged_axis", "source_normalized", "staged_normalized"),
    [
        ({}, {"description": None}, True, False),
        ({"description": None}, {}, False, True),
    ],
)
def test_axis_description_missing_and_null_are_symmetric(
    source_axis, staged_axis, source_normalized, staged_normalized
):
    comparison = equiv.require_definition_equivalent(
        _bytes({"axes": {"k": source_axis}}),
        _bytes({"axes": {"k": staged_axis}}),
    )
    assert comparison.normalized_pointers == ("/axes/k/description",)
    assert comparison.attestation["axis_description_states"]["k"] == {
        "source": "missing" if source_normalized else "null",
        "staged": "missing" if staged_normalized else "null",
        "source_normalized": source_normalized,
        "staged_normalized": staged_normalized,
    }


@pytest.mark.parametrize("container", equiv.TENSOR_SPEC_CONTAINERS)
@pytest.mark.parametrize(
    ("source_tensor", "staged_tensor", "source_normalized", "staged_normalized"),
    [
        ({}, {"description": None}, True, False),
        ({"description": None}, {}, False, True),
    ],
)
def test_direct_tensor_description_missing_and_null_are_symmetric(
    container,
    source_tensor,
    staged_tensor,
    source_normalized,
    staged_normalized,
):
    comparison = equiv.require_definition_equivalent(
        _bytes({container: {"x": source_tensor}}),
        _bytes({container: {"x": staged_tensor}}),
    )
    pointer = f"/{container}/x/description"
    assert comparison.normalized_pointers == (pointer,)
    assert comparison.attestation["tensor_container_states"][container] == {
        "source": "object",
        "staged": "object",
    }
    assert comparison.attestation["tensor_description_states"][pointer] == {
        "source": "missing" if source_normalized else "null",
        "staged": "missing" if staged_normalized else "null",
        "source_normalized": source_normalized,
        "staged_normalized": staged_normalized,
    }


def test_tensor_description_state_pointers_are_rfc6901_escaped_and_ordered():
    source = {
        "inputs": {"a/b": {}, "a~b": {}},
        "outputs": {"z": {}},
    }
    staged = {
        "inputs": {
            "a/b": {"description": None},
            "a~b": {"description": None},
        },
        "outputs": {"z": {"description": None}},
    }
    comparison = equiv.require_definition_equivalent(
        _bytes(source), _bytes(staged)
    )
    expected = (
        "/inputs/a~1b/description",
        "/inputs/a~0b/description",
        "/outputs/z/description",
    )
    assert comparison.normalized_pointers == expected
    assert tuple(comparison.tensor_description_states) == expected
    serialized = equiv.serialize_attestation(comparison.attestation)
    assert serialized.index(b'"/inputs/a~1b/description"') < serialized.index(
        b'"/inputs/a~0b/description"'
    )
    assert serialized.index(b'"/inputs/a~0b/description"') < serialized.index(
        b'"/outputs/z/description"'
    )


def test_normalized_pointer_order_is_axes_then_inputs_then_outputs():
    source = {
        "axes": {"z": {}},
        "inputs": {"z": {}},
        "outputs": {"z": {}},
    }
    staged = {
        "axes": {"z": {"description": None}},
        "inputs": {"z": {"description": None}},
        "outputs": {"z": {"description": None}},
    }
    comparison = equiv.require_definition_equivalent(
        _bytes(source), _bytes(staged)
    )
    assert comparison.normalized_pointers == (
        "/axes/z/description",
        "/inputs/z/description",
        "/outputs/z/description",
    )


@pytest.mark.parametrize(
    ("source_axis", "staged_axis", "normalized"),
    [
        ({}, {}, True),
        ({"description": None}, {"description": None}, False),
    ],
)
def test_equal_missing_or_null_axis_descriptions(source_axis, staged_axis, normalized):
    comparison = equiv.require_definition_equivalent(
        _bytes({"axes": {"k": source_axis}}),
        _bytes({"axes": {"k": staged_axis}}),
    )
    assert comparison.normalized_pointers == (
        ("/axes/k/description",) if normalized else ()
    )


def test_equal_explicit_description_and_unicode_escape_are_strict_by_value():
    explicit = {"axes": {"k": {"description": "measured axis"}}}
    assert equiv.require_definition_equivalent(_bytes(explicit), _bytes(explicit)).equal
    escaped = b'{"axes":{"k":{"description":"\\u00e9"}}}'
    decoded = '{"axes":{"k":{"description":"é"}}}'.encode("utf-8")
    assert equiv.require_definition_equivalent(escaped, decoded).equal


def test_axis_keys_are_rfc6901_escaped_in_states_and_pointers():
    source = {"axes": {"a/b": {}, "a~b": {}}}
    staged = {
        "axes": {
            "a/b": {"description": None},
            "a~b": {"description": None},
        }
    }
    comparison = equiv.require_definition_equivalent(_bytes(source), _bytes(staged))
    assert tuple(comparison.attestation["axis_description_states"]) == (
        "a~1b",
        "a~0b",
    )
    assert comparison.normalized_pointers == (
        "/axes/a~1b/description",
        "/axes/a~0b/description",
    )
    serialized = equiv.serialize_attestation(comparison.attestation)
    assert serialized.index(b'"a~1b"') < serialized.index(b'"a~0b"')


def test_mixed_axis_states_cover_sorted_union_and_normalization_flags():
    source = {
        "axes": {
            "a": {},
            "b": {"description": None},
            "c": {"description": "explicit"},
            "d": {"description": ""},
            "e": 7,
        }
    }
    staged = {
        "axes": {
            "a": {"description": None},
            "b": {},
            "c": {"description": "explicit"},
            "d": {"description": ""},
            "e": 7,
        }
    }
    comparison = equiv.require_definition_equivalent(_bytes(source), _bytes(staged))
    states = comparison.attestation["axis_description_states"]
    assert list(states) == ["a", "b", "c", "d", "e"]
    assert states == {
        "a": {
            "source": "missing",
            "staged": "null",
            "source_normalized": True,
            "staged_normalized": False,
        },
        "b": {
            "source": "null",
            "staged": "missing",
            "source_normalized": False,
            "staged_normalized": True,
        },
        "c": {
            "source": "nonempty_string",
            "staged": "nonempty_string",
            "source_normalized": False,
            "staged_normalized": False,
        },
        "d": {
            "source": "empty_string",
            "staged": "empty_string",
            "source_normalized": False,
            "staged_normalized": False,
        },
        "e": {
            "source": "axis_not_object",
            "staged": "axis_not_object",
            "source_normalized": False,
            "staged_normalized": False,
        },
    }


@pytest.mark.parametrize("axes", [object(), [], "axes", 7, None, True])
def test_identical_missing_or_nonobject_axes_are_equal(axes):
    if type(axes) is object:
        source = staged = {"name": "definition"}
        expected_state = "missing"
    else:
        source = staged = {"axes": axes}
        expected_state = "non_object"
    comparison = equiv.require_definition_equivalent(_bytes(source), _bytes(staged))
    assert comparison.axes_container_state == {
        "source": expected_state,
        "staged": expected_state,
    }
    assert comparison.axis_description_states == {}


@pytest.mark.parametrize(
    ("source_description", "staged_axis"),
    [
        ("explicit", {"description": None}),
        ("", {"description": None}),
        ("", {}),
        (True, {"description": None}),
        (1, {"description": None}),
        (1.0, {"description": None}),
        ([], {"description": None}),
        ({}, {"description": None}),
    ],
)
def test_ineligible_axis_description_values_never_normalize(
    source_description, staged_axis
):
    source = _bytes({"axes": {"k": {"description": source_description}}})
    staged = _bytes({"axes": {"k": staged_axis}})
    comparison = equiv.compare_definition_bytes(source, staged)
    assert comparison.difference_pointers == ("/axes/k/description",)
    assert equiv.compare_definition_bytes_v1(
        source, staged
    ).difference_pointers == comparison.difference_pointers


@pytest.mark.parametrize("container", equiv.TENSOR_SPEC_CONTAINERS)
@pytest.mark.parametrize(
    "source_description",
    ["explicit", "", True, 1, 1.0, [], {}],
)
def test_ineligible_tensor_description_values_never_normalize(
    container, source_description
):
    source = _bytes(
        {container: {"x": {"description": source_description}}}
    )
    staged = _bytes({container: {"x": {"description": None}}})
    comparison = equiv.compare_definition_bytes(source, staged)
    pointer = f"/{container}/x/description"
    assert comparison.difference_pointers == (pointer,)
    assert equiv.compare_definition_bytes_v1(
        source, staged
    ).difference_pointers == comparison.difference_pointers


@pytest.mark.parametrize("container", equiv.TENSOR_SPEC_CONTAINERS)
@pytest.mark.parametrize("bad_container", [[], "value", 1, None, True])
def test_nonobject_tensor_container_disables_eligibility(
    container, bad_container
):
    source = _bytes({container: bad_container})
    staged = _bytes({container: {"x": {"description": None}}})
    comparison = equiv.compare_definition_bytes(source, staged)
    assert comparison.difference_pointers == (f"/{container}",)
    assert comparison.tensor_container_states[container] == {
        "source": "non_object",
        "staged": "object",
    }
    state = comparison.tensor_description_states[
        f"/{container}/x/description"
    ]
    assert state["source"] == "tensor_not_object"
    assert state["source_normalized"] is False


@pytest.mark.parametrize("container", equiv.TENSOR_SPEC_CONTAINERS)
@pytest.mark.parametrize("bad_tensor", [[], "value", 1, None, True])
def test_nonobject_tensor_value_disables_eligibility(container, bad_tensor):
    source = _bytes({container: {"x": bad_tensor}})
    staged = _bytes({container: {"x": {"description": None}}})
    comparison = equiv.compare_definition_bytes(source, staged)
    assert comparison.difference_pointers == (f"/{container}/x",)
    state = comparison.tensor_description_states[
        f"/{container}/x/description"
    ]
    assert state["source"] == "tensor_not_object"
    assert state["source_normalized"] is False


@pytest.mark.parametrize(
    ("source", "staged", "pointer"),
    [
        (
            {"axes": {"k": {"nested": {}}}},
            {"axes": {"k": {"nested": {"description": None}}}},
            "/axes/k/nested/description",
        ),
        (
            {"inputs": {"x": {"nested": {}}}},
            {"inputs": {"x": {"nested": {"description": None}}}},
            "/inputs/x/nested/description",
        ),
        ({}, {"description": None}, "/description"),
        (
            {"wrapper": {"axes": {"k": {}}}},
            {"wrapper": {"axes": {"k": {"description": None}}}},
            "/wrapper/axes/k/description",
        ),
        ({"unknown": "present"}, {}, "/unknown"),
        ({"value": True}, {"value": 1}, "/value"),
        ({"value": 1}, {"value": 1.0}, "/value"),
    ],
)
def test_every_position_outside_the_closed_classes_remains_strict(
    source, staged, pointer
):
    source_bytes = _bytes(source)
    staged_bytes = _bytes(staged)
    comparison = equiv.compare_definition_bytes(source_bytes, staged_bytes)
    assert comparison.difference_pointers == (pointer,)
    assert equiv.compare_definition_bytes_v1(
        source_bytes, staged_bytes
    ).difference_pointers == comparison.difference_pointers
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.require_definition_equivalent(_bytes(source), _bytes(staged))
    assert caught.value.code == "sol_stage_source_mismatch"
    assert pointer in caught.value.detail


@pytest.mark.parametrize("staged_axes", [[], "axes", 1, None, True])
def test_object_vs_nonobject_axes_mismatches_at_container(staged_axes):
    source = _bytes({"axes": {"k": {"description": None}}})
    staged = _bytes({"axes": staged_axes})
    comparison = equiv.compare_definition_bytes(source, staged)
    assert comparison.difference_pointers == ("/axes",)
    assert comparison.axis_description_states["k"]["staged"] == "axis_not_object"
    assert equiv.compare_definition_bytes_v1(
        source, staged
    ).difference_pointers == comparison.difference_pointers


@pytest.mark.parametrize("source_axis", [[], "axis", 1, None, True])
def test_nonobject_axis_value_disables_eligibility(source_axis):
    source = _bytes({"axes": {"k": source_axis}})
    staged = _bytes({"axes": {"k": {"description": None}}})
    comparison = equiv.compare_definition_bytes(source, staged)
    assert comparison.difference_pointers == ("/axes/k",)
    assert comparison.axis_description_states["k"]["source"] == "axis_not_object"
    assert equiv.compare_definition_bytes_v1(
        source, staged
    ).difference_pointers == comparison.difference_pointers


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b'{"outer":{"key":1,"key":2}}', "sol_definition_duplicate_key"),
        (b'{"value":NaN}', "sol_definition_nonfinite"),
        (b'{"value":Infinity}', "sol_definition_nonfinite"),
        (b'{"value":-Infinity}', "sol_definition_nonfinite"),
        (b'{not-json}', "sol_definition_parse_error"),
        (b'\xff', "sol_definition_parse_error"),
        (b'[]', "sol_definition_parse_error"),
    ],
)
@pytest.mark.parametrize("bad_side", ["source", "staged"])
def test_strict_parse_refusals_are_typed(payload, code, bad_side):
    source = payload if bad_side == "source" else b"{}"
    staged = payload if bad_side == "staged" else b"{}"
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.compare_definition_bytes(source, staged)
    assert caught.value.code == code
    assert bad_side in caught.value.detail


def test_comparator_returns_every_difference_but_refusal_reports_first_twenty():
    source = {"items": {f"k{index:02d}": 0 for index in range(25)}}
    staged = {"items": {f"k{index:02d}": 1 for index in range(25)}}
    comparison = _comparison(source, staged)
    expected = tuple(f"/items/k{index:02d}" for index in range(25))
    assert comparison.difference_pointers == expected
    assert comparison.difference_count == 25

    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.require_definition_equivalent(_bytes(source), _bytes(staged))
    assert caught.value.code == "sol_stage_source_mismatch"
    assert caught.value.detail == (
        "definition differs at 25 JSON pointer(s); first 20: "
        + ", ".join(expected[:20])
    )


def test_attestation_serialization_is_deterministic_utf8_and_trailing_lf():
    comparison = equiv.require_definition_equivalent(
        '{"axes":{"é/":{"description":null}}}'.encode(),
        '{"axes":{"é/":{"description":null}}}'.encode(),
    )
    first = equiv.serialize_attestation(comparison.attestation)
    second = equiv.serialize_attestation(copy.deepcopy(comparison.attestation))
    assert first == second
    assert first.endswith(b"\n")
    assert b"\\u00e9" not in first
    assert json.loads(first) == comparison.attestation


def test_attestation_validator_has_distinct_rule_state_and_pointer_refusals():
    attestation = equiv.require_definition_equivalent(b"{}", b"{}").attestation

    wrong_rule = copy.deepcopy(attestation)
    wrong_rule["rule_id"] = "SOL_DEF_EQUIV_V1"
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.validate_attestation_shape_and_eligibility(wrong_rule)
    assert caught.value.code == "SOL_ATTESTATION_RULE_MISMATCH"

    wrong_state = copy.deepcopy(attestation)
    wrong_state["top_level_field_states"]["op_type"]["source_normalized"] = False
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.validate_attestation_shape_and_eligibility(wrong_state)
    assert caught.value.code == "SOL_ATTESTATION_STATE_MISMATCH"

    wrong_pointer = copy.deepcopy(attestation)
    wrong_pointer["axes_container_state"] = {"source": "object", "staged": "object"}
    wrong_pointer["axis_description_states"] = {
        "bad~escape": {
            "source": "null",
            "staged": "null",
            "source_normalized": False,
            "staged_normalized": False,
        }
    }
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.validate_attestation_shape_and_eligibility(wrong_pointer)
    assert caught.value.code == "SOL_ATTESTATION_POINTER_INELIGIBLE"

    wrong_tensor_pointer = copy.deepcopy(attestation)
    wrong_tensor_pointer["tensor_description_states"] = {
        "/wrapper/x/description": {
            "source": "null",
            "staged": "null",
            "source_normalized": False,
            "staged_normalized": False,
        }
    }
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.validate_attestation_shape_and_eligibility(
            wrong_tensor_pointer
        )
    assert caught.value.code == "SOL_ATTESTATION_POINTER_INELIGIBLE"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda value: value["top_level_field_states"]["op_type"].__setitem__(
                "source", []
            ),
            "SOL_ATTESTATION_STATE_MISMATCH",
        ),
        (
            lambda value: value["axes_container_state"].__setitem__(
                "source", []
            ),
            "SOL_ATTESTATION_STATE_MISMATCH",
        ),
        (
            lambda value: value["tensor_container_states"]["inputs"].__setitem__(
                "source", []
            ),
            "SOL_ATTESTATION_STATE_MISMATCH",
        ),
    ],
)
def test_attestation_nonstring_states_always_refuse_with_a_typed_error(
    mutate, code
):
    attestation = equiv.require_definition_equivalent(b"{}", b"{}").attestation
    mutate(attestation)
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.validate_attestation_shape_and_eligibility(attestation)
    assert caught.value.code == code


def test_normalization_beneath_nonobject_axis_is_pointer_ineligible():
    expected = equiv.require_definition_equivalent(
        _bytes({"axes": {"k": 7}}), _bytes({"axes": {"k": 7}})
    ).attestation
    forged = copy.deepcopy(expected)
    forged["axis_description_states"]["k"]["source"] = "missing"
    forged["axis_description_states"]["k"]["source_normalized"] = True

    assert equiv.ineligible_axis_state_pointers(forged, expected) == (
        "/axes/k/description",
    )


def test_tensor_normalization_outside_sealed_union_is_pointer_ineligible():
    expected = equiv.require_definition_equivalent(
        _bytes({"inputs": {"x": 7}}),
        _bytes({"inputs": {"x": 7}}),
    ).attestation
    forged = copy.deepcopy(expected)
    pointer = "/inputs/x/description"
    forged["tensor_description_states"][pointer]["source"] = "missing"
    forged["tensor_description_states"][pointer][
        "source_normalized"
    ] = True

    assert equiv.ineligible_normalized_state_pointers(
        forged, expected
    ) == (pointer,)


def test_extra_tensor_state_pointer_is_rejected_against_sealed_union():
    expected = equiv.require_definition_equivalent(b"{}", b"{}").attestation
    forged = copy.deepcopy(expected)
    pointer = "/outputs/unsealed/description"
    forged["tensor_container_states"]["outputs"] = {
        "source": "object",
        "staged": "object",
    }
    forged["tensor_description_states"][pointer] = {
        "source": "null",
        "staged": "null",
        "source_normalized": False,
        "staged_normalized": False,
    }
    assert equiv.ineligible_normalized_state_pointers(
        forged, expected
    ) == (pointer,)


def _e387_projection_fixture():
    expected = equiv.require_definition_equivalent(
        _bytes({"inputs": {"x": {}}}),
        _bytes({"inputs": {"x": {"description": None}}}),
    ).attestation
    observed = {
        key: copy.deepcopy(value)
        for key, value in expected.items()
        if key not in {
            "tensor_container_states", "tensor_description_states",
        }
    }
    return observed, expected


def test_generic_validator_rejects_seven_field_e387_projection():
    observed, _expected = _e387_projection_fixture()
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.validate_attestation_shape_and_eligibility(observed)
    assert caught.value.code == "SOL_ATTESTATION_STATE_MISMATCH"


def test_exact_e387_code_binding_accepts_only_current_minus_tensor_projection():
    observed, expected = _e387_projection_fixture()
    assert equiv.validate_e387_legacy_attestation_projection(
        observed,
        expected,
        source_commit=equiv.E387_LEGACY_SOURCE_COMMIT,
        comparator_sha256=equiv.E387_LEGACY_COMPARATOR_SHA256,
        capture_adapter_sha256=equiv.E387_LEGACY_CAPTURE_ADAPTER_SHA256,
    ) == observed
    serialized = equiv.serialize_e387_legacy_attestation(
        observed,
        expected,
        source_commit=equiv.E387_LEGACY_SOURCE_COMMIT,
        comparator_sha256=equiv.E387_LEGACY_COMPARATOR_SHA256,
        capture_adapter_sha256=equiv.E387_LEGACY_CAPTURE_ADAPTER_SHA256,
    )
    assert json.loads(serialized) == observed
    assert serialized.endswith(b"\n")


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_commit": "0" * 40},
        {"comparator_sha256": "0" * 64},
        {"capture_adapter_sha256": "0" * 64},
    ],
)
def test_e387_projection_rejects_wrong_commit_or_code_hash(overrides):
    observed, expected = _e387_projection_fixture()
    context = {
        "source_commit": equiv.E387_LEGACY_SOURCE_COMMIT,
        "comparator_sha256": equiv.E387_LEGACY_COMPARATOR_SHA256,
        "capture_adapter_sha256": (
            equiv.E387_LEGACY_CAPTURE_ADAPTER_SHA256
        ),
        **overrides,
    }
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.validate_e387_legacy_attestation_projection(
            observed, expected, **context
        )
    assert caught.value.code == "SOL_ATTESTATION_STATE_MISMATCH"


@pytest.mark.parametrize("tamper", ["field", "state"])
def test_e387_projection_rejects_wrong_field_or_state(tamper):
    observed, expected = _e387_projection_fixture()
    if tamper == "field":
        observed["unknown"] = None
    else:
        observed["top_level_field_states"]["op_type"]["source"] = "null"
        observed["top_level_field_states"]["op_type"][
            "source_normalized"
        ] = False
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.validate_e387_legacy_attestation_projection(
            observed,
            expected,
            source_commit=equiv.E387_LEGACY_SOURCE_COMMIT,
            comparator_sha256=equiv.E387_LEGACY_COMPARATOR_SHA256,
            capture_adapter_sha256=(
                equiv.E387_LEGACY_CAPTURE_ADAPTER_SHA256
            ),
        )
    assert caught.value.code == "SOL_ATTESTATION_STATE_MISMATCH"


def test_huge_integer_parse_failure_is_typed():
    payload = b'{"value":' + (b"9" * 5000) + b"}"
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        equiv.compare_definition_bytes(payload, b"{}")
    assert caught.value.code == "sol_definition_parse_error"


def test_frozen_v1_rejects_axis_default_materialization_while_v2_accepts_it():
    source = _bytes({"axes": {"k": {}}})
    staged = _bytes({"axes": {"k": {"description": None}}})
    v1 = equiv.compare_definition_bytes_v1(source, staged)
    v2 = equiv.compare_definition_bytes(source, staged)
    assert v1.rule_id == "SOL_DEF_EQUIV_V1"
    assert v1.difference_pointers == ("/axes/k/description",)
    assert not v1.equal
    assert v2.equal
    with pytest.raises(equiv.SolDefinitionEquivalenceError) as caught:
        _ = v1.attestation
    assert caught.value.code == "SOL_ATTESTATION_RULE_MISMATCH"

    tensor_source = _bytes({"inputs": {"x": {}}})
    tensor_staged = _bytes(
        {"inputs": {"x": {"description": None}}}
    )
    assert equiv.compare_definition_bytes_v1(
        tensor_source, tensor_staged
    ).difference_pointers == ("/inputs/x/description",)
    tensor_v2 = equiv.compare_definition_bytes(
        tensor_source, tensor_staged
    )
    assert tensor_v2.equal
    assert tensor_v2.normalized_pointers == (
        "/inputs/x/description",
    )

    unknown_source = _bytes({"unknown": 1})
    unknown_staged = _bytes({"unknown": 2})
    assert (
        equiv.compare_definition_bytes_v1(unknown_source, unknown_staged).difference_pointers
        == equiv.compare_definition_bytes(unknown_source, unknown_staged).difference_pointers
        == ("/unknown",)
    )
