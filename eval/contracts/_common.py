"""Shared target-free evaluator contract primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from tools.local_labels.immutable_artifact import ImmutableArtifactError


class AuthorityBlockerCode(str, Enum):
    """Closed reasons why a production evaluator boundary cannot proceed."""

    RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED = (
        "RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED"
    )
    LAUNCH_PARSER_AUTHORITY_UNAVAILABLE = "LAUNCH_PARSER_AUTHORITY_UNAVAILABLE"
    CROSS_NAMESPACE_PROVENANCE_AUTHORITY_UNAVAILABLE = (
        "CROSS_NAMESPACE_PROVENANCE_AUTHORITY_UNAVAILABLE"
    )
    BINDING_AUTHORITY_UNAVAILABLE = "BINDING_AUTHORITY_UNAVAILABLE"
    STATE_EVIDENCE_AUTHORITY_UNAVAILABLE = "STATE_EVIDENCE_AUTHORITY_UNAVAILABLE"
    TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE = (
        "TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE"
    )
    ANCESTRY_REVIEW_AUTHORITY_UNAVAILABLE = (
        "ANCESTRY_REVIEW_AUTHORITY_UNAVAILABLE"
    )
    COHORT_AUTHORITY_UNAVAILABLE = "COHORT_AUTHORITY_UNAVAILABLE"
    SIMULATOR_OUTPUT_AUTHORITY_UNAVAILABLE = (
        "SIMULATOR_OUTPUT_AUTHORITY_UNAVAILABLE"
    )
    LABEL_ADMISSION_AUTHORITY_UNAVAILABLE = (
        "LABEL_ADMISSION_AUTHORITY_UNAVAILABLE"
    )
    REVIEW_OPERATOR_TRUST_AUTHORITY_UNAVAILABLE = (
        "REVIEW_OPERATOR_TRUST_AUTHORITY_UNAVAILABLE"
    )


class EvaluatorAuthorityUnavailable(ImmutableArtifactError):
    """A typed fail-closed production boundary with no installed authority."""

    def __init__(self, code: AuthorityBlockerCode, detail: str) -> None:
        if not isinstance(code, AuthorityBlockerCode):
            raise ImmutableArtifactError("authority blocker code is invalid")
        if type(detail) is not str or not detail:
            raise ImmutableArtifactError("authority blocker detail is invalid")
        self.code = code
        super().__init__(f"{code.value}: {detail}")


def authority_unavailable(
    code: AuthorityBlockerCode, detail: str
) -> EvaluatorAuthorityUnavailable:
    """Construct one stable typed production blocker."""

    return EvaluatorAuthorityUnavailable(code, detail)


class OutcomeStatus(str, Enum):
    """The exhaustive status space for offline contract evaluation."""

    ADMITTED = "ADMITTED"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ContractOutcome:
    """One deterministic, total evaluator outcome."""

    status: OutcomeStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, OutcomeStatus):
            raise ImmutableArtifactError("outcome status is invalid")
        if type(self.reason_codes) is not tuple:
            raise ImmutableArtifactError("outcome reason_codes must be an exact tuple")
        if any(type(reason) is not str or not reason for reason in self.reason_codes):
            raise ImmutableArtifactError("outcome reason codes must be nonempty strings")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ImmutableArtifactError("duplicate outcome reason code")
        if self.status is OutcomeStatus.ADMITTED and self.reason_codes:
            raise ImmutableArtifactError("admitted outcome cannot carry failure reasons")
        if self.status is not OutcomeStatus.ADMITTED and not self.reason_codes:
            raise ImmutableArtifactError("non-admitted outcome requires a reason")

    def to_model(self) -> dict[str, Any]:
        return {"status": self.status.value, "reason_codes": list(self.reason_codes)}


_UNRESOLVED_SCIENTIFIC_INPUTS = (
    "CYCLE_MAPPING",
    "CYCLE_COMPOSITION",
    "COHORT",
    "PROTOCOL_NUMERIC_SETTINGS",
    "COVERAGE",
    "TRUST",
    "OPERATOR_APPROVAL",
)


def evaluation_boundary_model() -> dict[str, Any]:
    """Return fixed statuses without choosing any unresolved scientific input."""

    return {
        "mode": "TARGET_FREE_OFFLINE_ONLY",
        "m5": {
            "status": "M5_NOT_STARTED",
            "collection_status": "M5_COLLECT_DISABLED",
        },
        "scientific_inputs": [
            {"name": name, "status": "UNRESOLVED"}
            for name in _UNRESOLVED_SCIENTIFIC_INPUTS
        ],
        "cycles": {
            "metric_id": "gpc__cycles_elapsed.max",
            "implementation_status": "required_unimplemented",
            "evaluation_state": "pending",
            "outcome": "unavailable",
            "score_status": "unscored",
        },
    }


def require_exact_evaluation_boundary(value: object) -> None:
    if value != evaluation_boundary_model():
        raise ImmutableArtifactError("evaluation boundary is not the exact offline boundary")


def aggregate_outcomes(outcomes: Iterable[ContractOutcome]) -> ContractOutcome:
    """Aggregate deterministically with REJECTED taking fail-closed precedence."""

    material = tuple(outcomes)
    rejected = tuple(
        reason
        for outcome in material
        if outcome.status is OutcomeStatus.REJECTED
        for reason in outcome.reason_codes
    )
    if rejected:
        return ContractOutcome(OutcomeStatus.REJECTED, _deduplicate(rejected))
    unavailable = tuple(
        reason
        for outcome in material
        if outcome.status is OutcomeStatus.UNAVAILABLE
        for reason in outcome.reason_codes
    )
    if unavailable:
        return ContractOutcome(OutcomeStatus.UNAVAILABLE, _deduplicate(unavailable))
    return ContractOutcome(OutcomeStatus.ADMITTED, ())


def outcome_from_model(
    value: object,
    *,
    rejected_reasons: frozenset[str],
    unavailable_reasons: frozenset[str],
) -> ContractOutcome:
    """Parse an outcome while keeping rejected/unavailable vocabularies disjoint."""

    if type(value) is not dict or set(value) != {"status", "reason_codes"}:
        raise ImmutableArtifactError("outcome fields are not exact")
    try:
        status = OutcomeStatus(value["status"])
    except (TypeError, ValueError) as exc:
        raise ImmutableArtifactError("outcome status is invalid") from exc
    reasons = value["reason_codes"]
    if type(reasons) is not list:
        raise ImmutableArtifactError("outcome reason_codes must be an array")
    outcome = ContractOutcome(status, tuple(reasons))
    if status is OutcomeStatus.ADMITTED:
        vocabulary = frozenset()
    elif status is OutcomeStatus.REJECTED:
        vocabulary = rejected_reasons
    else:
        vocabulary = unavailable_reasons
    if any(reason not in vocabulary for reason in outcome.reason_codes):
        raise ImmutableArtifactError("outcome reason vocabulary does not match status")
    return outcome


def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    observed: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in observed:
            observed.add(value)
            result.append(value)
    return tuple(result)
