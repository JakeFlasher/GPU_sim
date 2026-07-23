#!/usr/bin/env python3
"""Closed equivalence contracts for SOL source and packaged definitions.

``SOL_DEF_EQUIV_V2`` is the only certifying rule.  V1 is retained solely as a
frozen, non-certifying reference for the preserved blocking audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Final, Mapping


NONCERTIFYING_V1_RULE_ID: Final = "SOL_DEF_EQUIV_V1"
CERTIFYING_RULE_ID: Final = "SOL_DEF_EQUIV_V2"
RULE_ID: Final = CERTIFYING_RULE_ID
E387_LEGACY_SOURCE_COMMIT: Final = (
    "e38788ca0a4457570bd562ea1389b07b3f855e22"
)
E387_LEGACY_COMPARATOR_SHA256: Final = (
    "573a8a84196da4c8105c1a1b742cde10bb9c1923a19fd1a056392c4eb78c9bfc"
)
E387_LEGACY_CAPTURE_ADAPTER_SHA256: Final = (
    "35672d28ddfcec9c9e2a9eb1021d7fc7c84b458e1865406ff0b4aa349a45d003"
)
OPTIONAL_TOP_LEVEL_FIELDS: Final = (
    "hf_id",
    "op_type",
    "custom_inputs_entrypoint",
)
TENSOR_SPEC_CONTAINERS: Final = ("inputs", "outputs")
MAX_REFUSAL_DIFFERENCE_POINTERS: Final = 20
# Compatibility name for callers that only use the refusal-payload bound.
MAX_DIFFERENCE_POINTERS: Final = MAX_REFUSAL_DIFFERENCE_POINTERS

_GENERIC_STATES = frozenset(
    {"missing", "null", "empty_string", "nonempty_string", "non_string"}
)
_AXIS_STATES = _GENERIC_STATES | {"axis_not_object"}
_TENSOR_STATES = _GENERIC_STATES | {"tensor_not_object"}
_AXES_CONTAINER_STATES = frozenset({"object", "missing", "non_object"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MISSING = object()
_CURRENT_ATTESTATION_FIELDS = frozenset({
    "rule_id",
    "result",
    "source_definition_sha256",
    "staged_definition_sha256",
    "top_level_field_states",
    "axes_container_state",
    "axis_description_states",
    "tensor_container_states",
    "tensor_description_states",
})
_E387_LEGACY_ATTESTATION_FIELDS = _CURRENT_ATTESTATION_FIELDS - {
    "tensor_container_states",
    "tensor_description_states",
}


class SolDefinitionEquivalenceError(ValueError):
    """Typed refusal raised while parsing, comparing, or validating evidence."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str):
        super().__init__(key)
        self.key = key


class _NonfiniteTokenError(ValueError):
    def __init__(self, token: str):
        super().__init__(token)
        self.token = token


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(key)
        value[key] = item
    return value


def _reject_nonfinite(token: str) -> Any:
    raise _NonfiniteTokenError(token)


def parse_definition_bytes(data: bytes, *, side: str) -> dict[str, Any]:
    """Parse one UTF-8 JSON object without duplicate keys or nonfinite values."""

    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except _DuplicateKeyError as exc:
        raise SolDefinitionEquivalenceError(
            "sol_definition_duplicate_key",
            f"{side} definition contains duplicate key {exc.key!r}",
        ) from exc
    except _NonfiniteTokenError as exc:
        raise SolDefinitionEquivalenceError(
            "sol_definition_nonfinite",
            f"{side} definition contains nonfinite token {exc.token!r}",
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SolDefinitionEquivalenceError(
            "sol_definition_parse_error",
            f"cannot parse {side} definition as UTF-8 JSON: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise SolDefinitionEquivalenceError(
            "sol_definition_parse_error",
            f"{side} definition top-level value must be a JSON object",
        )
    return value


def representation_state(document: Mapping[str, Any], field: str) -> str:
    """Return the closed representation state for one object member."""

    value = document.get(field, _MISSING)
    if value is _MISSING:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "empty_string" if value == "" else "nonempty_string"
    return "non_string"


def _canonicalize_top_level(document: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(document)
    for field in OPTIONAL_TOP_LEVEL_FIELDS:
        value = canonical.get(field, _MISSING)
        if (
            value is _MISSING
            or value is None
            or (isinstance(value, str) and value == "")
        ):
            canonical[field] = None
    return canonical


def canonicalize_definition_v1(document: dict[str, Any]) -> dict[str, Any]:
    """Frozen V1 canonicalization; never use it to certify new evidence."""

    return _canonicalize_top_level(document)


def canonicalize_definition(document: dict[str, Any]) -> dict[str, Any]:
    """Apply the closed V2 representation-only normalizations.

    Pydantic's ``Definition.model_dump_json()`` materializes a missing direct
    ``TensorSpec.description`` as JSON null.  That representation change is
    eligible only for immediate objects in the top-level ``inputs`` and
    ``outputs`` maps.  Container type changes, non-object tensor entries, and
    every deeper or displaced description remain strict.
    """

    canonical = _canonicalize_top_level(document)
    axes = canonical.get("axes", _MISSING)
    if isinstance(axes, dict):
        canonical_axes = dict(axes)
        canonical["axes"] = canonical_axes
        for axis_key, axis_value in axes.items():
            if not isinstance(axis_value, dict):
                continue
            canonical_axis = dict(axis_value)
            canonical_axes[axis_key] = canonical_axis
            if "description" not in canonical_axis:
                canonical_axis["description"] = None

    for container_name in TENSOR_SPEC_CONTAINERS:
        container = canonical.get(container_name, _MISSING)
        if not isinstance(container, dict):
            continue
        canonical_container = dict(container)
        canonical[container_name] = canonical_container
        for tensor_key, tensor_value in container.items():
            if not isinstance(tensor_value, dict):
                continue
            canonical_tensor = dict(tensor_value)
            canonical_container[tensor_key] = canonical_tensor
            if "description" not in canonical_tensor:
                canonical_tensor["description"] = None
    return canonical


def _escape_pointer_component(component: str | int) -> str:
    return str(component).replace("~", "~0").replace("/", "~1")


def _unescape_pointer_component(component: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(component):
        char = component[index]
        if char != "~":
            output.append(char)
            index += 1
            continue
        if index + 1 >= len(component) or component[index + 1] not in "01":
            raise ValueError("invalid RFC 6901 escape")
        output.append("~" if component[index + 1] == "0" else "/")
        index += 2
    return "".join(output)


def _pointer_child(parent: str, component: str | int) -> str:
    return f"{parent}/{_escape_pointer_component(component)}"


def _difference_pointers(
    source: Any,
    staged: Any,
    *,
    path: str,
    output: list[str],
) -> None:
    # JSON bool is a Python int subclass, so exact runtime types are part of
    # the contract instead of relying on ordinary Python equality.
    if type(source) is not type(staged):
        output.append(path)
        return

    if isinstance(source, dict):
        source_keys = set(source)
        staged_keys = set(staged)
        for key in sorted(source_keys | staged_keys):
            child = _pointer_child(path, key)
            if key not in source_keys or key not in staged_keys:
                output.append(child)
            else:
                _difference_pointers(
                    source[key], staged[key], path=child, output=output
                )
        return

    if isinstance(source, list):
        for index in range(max(len(source), len(staged))):
            child = _pointer_child(path, index)
            if index >= len(source) or index >= len(staged):
                output.append(child)
            else:
                _difference_pointers(
                    source[index], staged[index], path=child, output=output
                )
        return

    if source != staged:
        output.append(path)


def _axes_container_state(document: Mapping[str, Any]) -> str:
    axes = document.get("axes", _MISSING)
    if axes is _MISSING:
        return "missing"
    return "object" if isinstance(axes, dict) else "non_object"


def _container_state(document: Mapping[str, Any], field: str) -> str:
    value = document.get(field, _MISSING)
    if value is _MISSING:
        return "missing"
    return "object" if isinstance(value, dict) else "non_object"


def _top_level_field_states(
    source: Mapping[str, Any], staged: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for field in OPTIONAL_TOP_LEVEL_FIELDS:
        source_state = representation_state(source, field)
        staged_state = representation_state(staged, field)
        result[field] = {
            "source": source_state,
            "staged": staged_state,
            "source_normalized": source_state in {"missing", "empty_string"},
            "staged_normalized": staged_state in {"missing", "empty_string"},
        }
    return result


def _axis_description_state(axes: Any, axis_key: str) -> str:
    if not isinstance(axes, dict):
        return "axis_not_object"
    axis_value = axes.get(axis_key, _MISSING)
    if not isinstance(axis_value, dict):
        return "axis_not_object"
    return representation_state(axis_value, "description")


def _axis_description_states(
    source: Mapping[str, Any], staged: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    source_axes = source.get("axes", _MISSING)
    staged_axes = staged.get("axes", _MISSING)
    keys: set[str] = set()
    if isinstance(source_axes, dict):
        keys.update(source_axes)
    if isinstance(staged_axes, dict):
        keys.update(staged_axes)
    result: dict[str, dict[str, Any]] = {}
    normalized_pointers: list[str] = []
    for axis_key in sorted(keys):
        source_state = _axis_description_state(source_axes, axis_key)
        staged_state = _axis_description_state(staged_axes, axis_key)
        source_normalized = source_state == "missing"
        staged_normalized = staged_state == "missing"
        escaped_key = _escape_pointer_component(axis_key)
        result[escaped_key] = {
            "source": source_state,
            "staged": staged_state,
            "source_normalized": source_normalized,
            "staged_normalized": staged_normalized,
        }
        if source_normalized or staged_normalized:
            normalized_pointers.append(
                f"/axes/{escaped_key}/description"
            )
    return result, tuple(normalized_pointers)


def _tensor_description_state(container: Any, tensor_key: str) -> str:
    if not isinstance(container, dict):
        return "tensor_not_object"
    tensor_value = container.get(tensor_key, _MISSING)
    if not isinstance(tensor_value, dict):
        return "tensor_not_object"
    return representation_state(tensor_value, "description")


def _tensor_description_states(
    source: Mapping[str, Any], staged: Mapping[str, Any]
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
    tuple[str, ...],
]:
    container_states: dict[str, dict[str, str]] = {}
    description_states: dict[str, dict[str, Any]] = {}
    normalized_pointers: list[str] = []
    for container_name in TENSOR_SPEC_CONTAINERS:
        source_container = source.get(container_name, _MISSING)
        staged_container = staged.get(container_name, _MISSING)
        container_states[container_name] = {
            "source": _container_state(source, container_name),
            "staged": _container_state(staged, container_name),
        }
        keys: set[str] = set()
        if isinstance(source_container, dict):
            keys.update(source_container)
        if isinstance(staged_container, dict):
            keys.update(staged_container)
        for tensor_key in sorted(keys):
            source_state = _tensor_description_state(
                source_container, tensor_key
            )
            staged_state = _tensor_description_state(
                staged_container, tensor_key
            )
            source_normalized = source_state == "missing"
            staged_normalized = staged_state == "missing"
            pointer = (
                f"/{container_name}/"
                f"{_escape_pointer_component(tensor_key)}/description"
            )
            description_states[pointer] = {
                "source": source_state,
                "staged": staged_state,
                "source_normalized": source_normalized,
                "staged_normalized": staged_normalized,
            }
            if source_normalized or staged_normalized:
                normalized_pointers.append(pointer)
    return (
        container_states,
        description_states,
        tuple(normalized_pointers),
    )


@dataclass(frozen=True)
class DefinitionComparison:
    """A complete comparison result and its rederivable representation states."""

    rule_id: str
    source_sha256: str
    staged_sha256: str
    source_document: dict[str, Any]
    staged_document: dict[str, Any]
    canonical_source: dict[str, Any]
    canonical_staged: dict[str, Any]
    top_level_field_states: dict[str, dict[str, Any]]
    axes_container_state: dict[str, str]
    axis_description_states: dict[str, dict[str, Any]]
    tensor_container_states: dict[str, dict[str, str]]
    tensor_description_states: dict[str, dict[str, Any]]
    normalized_pointers: tuple[str, ...]
    difference_pointers: tuple[str, ...]

    @property
    def equal(self) -> bool:
        return not self.difference_pointers

    @property
    def difference_count(self) -> int:
        return len(self.difference_pointers)

    @property
    def field_states(self) -> dict[str, dict[str, Any]]:
        """Compatibility alias for the certifying top-level state map."""

        return self.top_level_field_states

    @property
    def attestation(self) -> dict[str, Any]:
        if self.rule_id != CERTIFYING_RULE_ID:
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_RULE_MISMATCH",
                f"{self.rule_id} is a frozen non-certifying rule",
            )
        if not self.equal:
            raise SolDefinitionEquivalenceError(
                "sol_stage_source_mismatch",
                _mismatch_detail(self.difference_pointers),
            )
        return {
            "rule_id": CERTIFYING_RULE_ID,
            "result": "equal",
            "source_definition_sha256": self.source_sha256,
            "staged_definition_sha256": self.staged_sha256,
            "top_level_field_states": self.top_level_field_states,
            "axes_container_state": self.axes_container_state,
            "axis_description_states": self.axis_description_states,
            "tensor_container_states": self.tensor_container_states,
            "tensor_description_states": self.tensor_description_states,
        }


def _compare_definition_bytes(
    source_bytes: bytes, staged_bytes: bytes, *, rule_id: str
) -> DefinitionComparison:
    source = parse_definition_bytes(source_bytes, side="source")
    staged = parse_definition_bytes(staged_bytes, side="staged")
    if rule_id == NONCERTIFYING_V1_RULE_ID:
        canonical_source = canonicalize_definition_v1(source)
        canonical_staged = canonicalize_definition_v1(staged)
    elif rule_id == CERTIFYING_RULE_ID:
        canonical_source = canonicalize_definition(source)
        canonical_staged = canonicalize_definition(staged)
    else:  # Internal programming error; callers cannot define another rule.
        raise ValueError(f"unsupported definition-equivalence rule: {rule_id}")
    pointers: list[str] = []
    _difference_pointers(
        canonical_source, canonical_staged, path="", output=pointers
    )
    axis_states, axis_normalized_pointers = _axis_description_states(
        source, staged
    )
    (
        tensor_container_states,
        tensor_states,
        tensor_normalized_pointers,
    ) = _tensor_description_states(source, staged)
    return DefinitionComparison(
        rule_id=rule_id,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        staged_sha256=hashlib.sha256(staged_bytes).hexdigest(),
        source_document=source,
        staged_document=staged,
        canonical_source=canonical_source,
        canonical_staged=canonical_staged,
        top_level_field_states=_top_level_field_states(source, staged),
        axes_container_state={
            "source": _axes_container_state(source),
            "staged": _axes_container_state(staged),
        },
        axis_description_states=axis_states,
        tensor_container_states=tensor_container_states,
        tensor_description_states=tensor_states,
        normalized_pointers=(
            axis_normalized_pointers + tensor_normalized_pointers
            if rule_id == CERTIFYING_RULE_ID
            else ()
        ),
        difference_pointers=tuple(pointers),
    )


def compare_definition_bytes(
    source_bytes: bytes, staged_bytes: bytes
) -> DefinitionComparison:
    """Compare exact bytes under the only certifying rule, V2."""

    return _compare_definition_bytes(
        source_bytes, staged_bytes, rule_id=CERTIFYING_RULE_ID
    )


def compare_definition_bytes_v1(
    source_bytes: bytes, staged_bytes: bytes
) -> DefinitionComparison:
    """Compare under frozen V1 semantics without producing attestations."""

    return _compare_definition_bytes(
        source_bytes, staged_bytes, rule_id=NONCERTIFYING_V1_RULE_ID
    )


def _mismatch_detail(pointers: tuple[str, ...]) -> str:
    first = pointers[:MAX_REFUSAL_DIFFERENCE_POINTERS]
    rendered = [pointer or "<root>" for pointer in first]
    return (
        f"definition differs at {len(pointers)} JSON pointer(s); "
        f"first {len(first)}: " + ", ".join(rendered)
    )


def require_definition_equivalent(
    source_bytes: bytes, staged_bytes: bytes
) -> DefinitionComparison:
    """Return a V2-equal comparison or raise the binding typed mismatch."""

    comparison = compare_definition_bytes(source_bytes, staged_bytes)
    if not comparison.equal:
        raise SolDefinitionEquivalenceError(
            "sol_stage_source_mismatch",
            _mismatch_detail(comparison.difference_pointers),
        )
    return comparison


def _state_record_valid(
    value: Any, *, kind: str
) -> bool:
    if kind == "axis":
        allowed = _AXIS_STATES
        normalized_states = {"missing"}
    elif kind == "tensor":
        allowed = _TENSOR_STATES
        normalized_states = {"missing"}
    elif kind == "top_level":
        allowed = _GENERIC_STATES
        normalized_states = {"missing", "empty_string"}
    else:
        raise ValueError(f"unknown representation-state kind: {kind}")
    if not isinstance(value, dict) or set(value) != {
        "source",
        "staged",
        "source_normalized",
        "staged_normalized",
    }:
        return False
    for side in ("source", "staged"):
        state = value.get(side)
        normalized = value.get(f"{side}_normalized")
        if (
            not isinstance(state, str)
            or state not in allowed
            or type(normalized) is not bool
        ):
            return False
        expected = state in normalized_states
        if normalized is not expected:
            return False
    return True


def validate_attestation_shape_and_eligibility(
    attestation: Any,
) -> dict[str, Any]:
    """Validate the closed V2 attestation shape without consulting stage bytes."""

    if not isinstance(attestation, dict) or attestation.get("rule_id") != RULE_ID:
        observed = attestation.get("rule_id") if isinstance(attestation, dict) else None
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_RULE_MISMATCH",
            f"expected {RULE_ID!r}, observed {observed!r}",
        )
    if (
        set(attestation) != _CURRENT_ATTESTATION_FIELDS
        or attestation.get("result") != "equal"
    ):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "attestation has the wrong fields or non-equal result",
        )
    for field in ("source_definition_sha256", "staged_definition_sha256"):
        value = attestation.get(field)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_STATE_MISMATCH", f"invalid {field}"
            )
    top_states = attestation.get("top_level_field_states")
    if (
        not isinstance(top_states, dict)
        or set(top_states) != set(OPTIONAL_TOP_LEVEL_FIELDS)
        or any(
            not _state_record_valid(value, kind="top_level")
            for value in top_states.values()
        )
    ):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "top_level_field_states violates the closed state contract",
        )
    container = attestation.get("axes_container_state")
    if (
        not isinstance(container, dict)
        or set(container) != {"source", "staged"}
        or any(
            not isinstance(value, str) or value not in _AXES_CONTAINER_STATES
            for value in container.values()
        )
    ):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "axes_container_state violates the closed state contract",
        )
    axis_states = attestation.get("axis_description_states")
    if not isinstance(axis_states, dict):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "axis_description_states must be an object",
        )
    for escaped_key, state_record in axis_states.items():
        if not isinstance(escaped_key, str):
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_POINTER_INELIGIBLE",
                "axis-description state key is not a string",
            )
        try:
            decoded_key = _unescape_pointer_component(escaped_key)
        except ValueError as exc:
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_POINTER_INELIGIBLE",
                f"invalid escaped axis key {escaped_key!r}",
            ) from exc
        if _escape_pointer_component(decoded_key) != escaped_key:
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_POINTER_INELIGIBLE",
                f"non-canonical escaped axis key {escaped_key!r}",
            )
        if not _state_record_valid(state_record, kind="axis"):
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_STATE_MISMATCH",
                f"invalid state record for axis {escaped_key!r}",
            )
        for side in ("source", "staged"):
            if (
                container[side] != "object"
                and state_record[f"{side}_normalized"]
            ):
                raise SolDefinitionEquivalenceError(
                    "SOL_ATTESTATION_POINTER_INELIGIBLE",
                    f"axis {escaped_key!r} normalizes beneath non-object "
                    f"{side} axes",
                )
            if (
                container[side] != "object"
                and state_record[side] != "axis_not_object"
            ):
                raise SolDefinitionEquivalenceError(
                    "SOL_ATTESTATION_STATE_MISMATCH",
                    f"axis {escaped_key!r} cannot exist under {side} axes state",
                )
    if (
        container["source"] != "object"
        and container["staged"] != "object"
        and axis_states
    ):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "axis states exist while neither axes container is an object",
        )

    tensor_containers = attestation.get("tensor_container_states")
    if (
        not isinstance(tensor_containers, dict)
        or set(tensor_containers) != set(TENSOR_SPEC_CONTAINERS)
    ):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "tensor_container_states violates the closed container set",
        )
    for container_name, states in tensor_containers.items():
        if (
            not isinstance(states, dict)
            or set(states) != {"source", "staged"}
            or any(
                not isinstance(value, str)
                or value not in _AXES_CONTAINER_STATES
                for value in states.values()
            )
        ):
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_STATE_MISMATCH",
                f"invalid tensor container states for {container_name!r}",
            )

    tensor_states = attestation.get("tensor_description_states")
    if not isinstance(tensor_states, dict):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "tensor_description_states must be an object",
        )
    state_paths_by_container = {
        container_name: [] for container_name in TENSOR_SPEC_CONTAINERS
    }
    for pointer, state_record in tensor_states.items():
        if not isinstance(pointer, str):
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_POINTER_INELIGIBLE",
                "tensor-description state pointer is not a string",
            )
        components = pointer.split("/")
        if (
            len(components) != 4
            or components[0] != ""
            or components[1] not in TENSOR_SPEC_CONTAINERS
            or components[3] != "description"
        ):
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_POINTER_INELIGIBLE",
                f"ineligible tensor-description pointer {pointer!r}",
            )
        container_name = components[1]
        escaped_key = components[2]
        try:
            decoded_key = _unescape_pointer_component(escaped_key)
        except ValueError as exc:
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_POINTER_INELIGIBLE",
                f"invalid escaped tensor key in {pointer!r}",
            ) from exc
        if _escape_pointer_component(decoded_key) != escaped_key:
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_POINTER_INELIGIBLE",
                f"non-canonical escaped tensor key in {pointer!r}",
            )
        if not _state_record_valid(state_record, kind="tensor"):
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_STATE_MISMATCH",
                f"invalid state record for tensor pointer {pointer!r}",
            )
        state_paths_by_container[container_name].append(pointer)
        for side in ("source", "staged"):
            container_state = tensor_containers[container_name][side]
            if (
                container_state != "object"
                and state_record[f"{side}_normalized"]
            ):
                raise SolDefinitionEquivalenceError(
                    "SOL_ATTESTATION_POINTER_INELIGIBLE",
                    f"{pointer!r} normalizes beneath non-object {side} "
                    f"{container_name}",
                )
            if (
                container_state != "object"
                and state_record[side] != "tensor_not_object"
            ):
                raise SolDefinitionEquivalenceError(
                    "SOL_ATTESTATION_STATE_MISMATCH",
                    f"{pointer!r} cannot exist under {side} "
                    f"{container_name} state",
                )
    for container_name, pointers in state_paths_by_container.items():
        states = tensor_containers[container_name]
        if (
            states["source"] != "object"
            and states["staged"] != "object"
            and pointers
        ):
            raise SolDefinitionEquivalenceError(
                "SOL_ATTESTATION_STATE_MISMATCH",
                f"tensor states exist while neither {container_name} "
                "container is an object",
            )
    return attestation


def is_e387_legacy_attestation_shape(value: Any) -> bool:
    """Return whether ``value`` has exactly the pre-correction V2 field set.

    This is shape detection only.  It never authorizes the historical shape;
    authorization requires the exact commit/code pins and byte-derived
    projection in :func:`validate_e387_legacy_attestation_projection`.
    """

    return (
        isinstance(value, dict)
        and set(value) == _E387_LEGACY_ATTESTATION_FIELDS
    )


def validate_e387_legacy_attestation_projection(
    observed_attestation: Any,
    expected_current_attestation: Mapping[str, Any],
    *,
    source_commit: str,
    comparator_sha256: str,
    capture_adapter_sha256: str,
) -> dict[str, Any]:
    """Validate the sole seven-field V2 compatibility projection.

    The clean e387 capture code predates the two TensorSpec state fields.  Its
    compatibility shim changes only the canonical equality decision, so its
    attestation is exactly the current byte-derived attestation with those two
    fields removed.  This helper accepts that projection only under the exact
    source commit and exact comparator/adapter blobs that produced it.  New
    emission and the generic validator remain nine-field-only.
    """

    if (
        source_commit != E387_LEGACY_SOURCE_COMMIT
        or comparator_sha256 != E387_LEGACY_COMPARATOR_SHA256
        or capture_adapter_sha256 != E387_LEGACY_CAPTURE_ADAPTER_SHA256
    ):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "seven-field V2 attestation lacks the exact e387 code binding",
        )
    validate_attestation_shape_and_eligibility(
        dict(expected_current_attestation)
    )
    if not is_e387_legacy_attestation_shape(observed_attestation):
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "e387 V2 attestation has the wrong projected field set",
        )
    projected = {
        key: expected_current_attestation[key]
        for key in expected_current_attestation
        if key in _E387_LEGACY_ATTESTATION_FIELDS
    }
    if observed_attestation != projected:
        raise SolDefinitionEquivalenceError(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "e387 V2 attestation is not the exact current-minus-tensor projection",
        )
    return projected


def serialize_e387_legacy_attestation(
    observed_attestation: Any,
    expected_current_attestation: Mapping[str, Any],
    *,
    source_commit: str,
    comparator_sha256: str,
    capture_adapter_sha256: str,
) -> bytes:
    """Serialize one fully validated e387-only seven-field projection."""

    projected = validate_e387_legacy_attestation_projection(
        observed_attestation,
        expected_current_attestation,
        source_commit=source_commit,
        comparator_sha256=comparator_sha256,
        capture_adapter_sha256=capture_adapter_sha256,
    )
    ordered = _attestation_json_order(projected, path=())
    return (
        json.dumps(
            ordered,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def ineligible_axis_state_pointers(
    attestation: Mapping[str, Any], expected_attestation: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return recorded axis-state positions outside the byte-derived union.

    Both inputs must already satisfy the closed structural contract.  The
    expected attestation is rederived from the sealed source/staged bytes, so
    it is the authority for whether an axis value is an object on each side.
    """

    validate_attestation_shape_and_eligibility(attestation)
    validate_attestation_shape_and_eligibility(expected_attestation)
    observed_states = attestation["axis_description_states"]
    expected_states = expected_attestation["axis_description_states"]
    ineligible: list[str] = []
    for escaped_key in sorted(
        observed_states, key=lambda key: _unescape_pointer_component(key)
    ):
        pointer = f"/axes/{escaped_key}/description"
        expected = expected_states.get(escaped_key)
        if expected is None:
            ineligible.append(pointer)
            continue
        observed = observed_states[escaped_key]
        if any(
            observed[f"{side}_normalized"]
            and expected[side] == "axis_not_object"
            for side in ("source", "staged")
        ):
            ineligible.append(pointer)
    return tuple(ineligible)


def ineligible_normalized_state_pointers(
    attestation: Mapping[str, Any], expected_attestation: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return axis or TensorSpec state positions outside sealed eligibility."""

    ineligible = list(
        ineligible_axis_state_pointers(attestation, expected_attestation)
    )
    observed_states = attestation["tensor_description_states"]
    expected_states = expected_attestation["tensor_description_states"]
    for pointer in sorted(observed_states, key=_tensor_pointer_sort_key):
        expected = expected_states.get(pointer)
        if expected is None:
            ineligible.append(pointer)
            continue
        observed = observed_states[pointer]
        if any(
            observed[f"{side}_normalized"]
            and expected[side] == "tensor_not_object"
            for side in ("source", "staged")
        ):
            ineligible.append(pointer)
    return tuple(ineligible)


def _tensor_pointer_sort_key(pointer: str) -> tuple[int, str]:
    components = pointer.split("/")
    container_index = TENSOR_SPEC_CONTAINERS.index(components[1])
    return container_index, _unescape_pointer_component(components[2])


def _attestation_json_order(value: Any, *, path: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        if path == ("axis_description_states",):
            keys = sorted(value, key=lambda key: _unescape_pointer_component(key))
        elif path == ("tensor_description_states",):
            keys = sorted(value, key=_tensor_pointer_sort_key)
        else:
            keys = sorted(value)
        return {
            key: _attestation_json_order(value[key], path=(*path, key))
            for key in keys
        }
    if isinstance(value, list):
        return [
            _attestation_json_order(item, path=(*path, str(index)))
            for index, item in enumerate(value)
        ]
    return value


def serialize_attestation(attestation: Mapping[str, Any]) -> bytes:
    """Serialize one validated V2 attestation in its certifying byte form."""

    validate_attestation_shape_and_eligibility(attestation)
    ordered = _attestation_json_order(dict(attestation), path=())
    return (
        json.dumps(
            ordered,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "CERTIFYING_RULE_ID",
    "DefinitionComparison",
    "E387_LEGACY_CAPTURE_ADAPTER_SHA256",
    "E387_LEGACY_COMPARATOR_SHA256",
    "E387_LEGACY_SOURCE_COMMIT",
    "MAX_DIFFERENCE_POINTERS",
    "MAX_REFUSAL_DIFFERENCE_POINTERS",
    "NONCERTIFYING_V1_RULE_ID",
    "OPTIONAL_TOP_LEVEL_FIELDS",
    "TENSOR_SPEC_CONTAINERS",
    "RULE_ID",
    "SolDefinitionEquivalenceError",
    "canonicalize_definition",
    "canonicalize_definition_v1",
    "compare_definition_bytes",
    "compare_definition_bytes_v1",
    "ineligible_axis_state_pointers",
    "ineligible_normalized_state_pointers",
    "is_e387_legacy_attestation_shape",
    "parse_definition_bytes",
    "representation_state",
    "require_definition_equivalent",
    "serialize_attestation",
    "serialize_e387_legacy_attestation",
    "validate_e387_legacy_attestation_projection",
    "validate_attestation_shape_and_eligibility",
]
