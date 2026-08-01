"""Deterministic synthetic-fixture scoring for the frozen P2 evaluator contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from eval.contracts._common import (
    AuthorityBlockerCode,
    EvaluatorAuthorityUnavailable,
    authority_unavailable,
)
from eval.contracts.validation import (
    METRIC_KEYS,
    ContractValidationError,
    canonical_launch_key_token,
    canonical_sha256,
    loads_json_strict,
    validate_report,
    validate_result,
)
from eval.labels import (
    LabelMetric,
    LaunchIdentity,
    MatchedLabels,
    MatchedLaunch,
    known_synthetic_authority_pin,
    load_synthetic_matched_labels,
)
from eval.registry import (
    CYCLE_BLOCKER,
    CYCLE_METRIC_KEY,
    MetricDescriptor,
    MetricRegistry,
    load_metric_registry,
)

_MAX_RESULT_BYTES = 64 * 1024 * 1024
_MAX_BUILD_IDENTITY_BYTES = 64 * 1024
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BUILD_IDENTITY_VERSION = "synthetic_evaluator_build_identity.v1"
_SYNTHETIC_ROLE = "synthetic_fixture_only_not_production"
_BUILD_IDENTITY_MEMBERS = frozenset(
    {
        "schema_version",
        "identity_sha256",
        "authority_role",
        "fixture_id",
        "evaluator_git_commit",
        "evaluator_git_dirty",
    }
)
KNOWN_SYNTHETIC_BUILD_IDENTITY_FILE_SHA256 = MappingProxyType(
    {
        "evaluator-mvp-canary-v1": (
            "ef738357b7a07539957134652d623677ac30503be63c0b04b2afbcd835686e31"
        )
    }
)
_COUNT_KEYS = frozenset(METRIC_KEYS[:4])
_RATE_KEYS = frozenset(METRIC_KEYS[4:7])
_NOISE_REASON_ORDER = (
    "dispersion_unavailable",
    "rule_not_absolute_interpretable",
)
_UNAVAILABLE_ORDER = (
    ("unavailable", "mem_mode_lacks_unit", "l1tex", None, None),
    ("unavailable", "mem_mode_lacks_unit", "lts", None, None),
    ("unavailable", "mem_mode_lacks_unit", "dram", None, None),
    ("unavailable", "undefined_fraction", None, "zero_denominator", None),
    ("required_unimplemented", None, None, None, CYCLE_BLOCKER),
)
_CYCLE_UNAVAILABLE_KEY = _UNAVAILABLE_ORDER[-1]


class EvaluatorScoringError(ValueError):
    """Typed failure outside the already sealed P1 input validators."""

    def __init__(
        self, code: str, detail: str, *, context: Mapping[str, Any] | None = None
    ) -> None:
        self.code = code
        self.detail = detail
        self.context = dict(context or {})
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class EvaluatorBuildIdentity:
    evaluator_git_commit: str
    evaluator_git_dirty: bool
    fixture_id: str
    identity_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.evaluator_git_commit) is not str
            or _SHA1_RE.fullmatch(self.evaluator_git_commit) is None
        ):
            raise EvaluatorScoringError(
                "EVALUATOR_BUILD_IDENTITY_INVALID",
                "evaluator Git commit is not 40 lowercase hexadecimal characters",
            )
        if type(self.evaluator_git_dirty) is not bool:
            raise EvaluatorScoringError(
                "EVALUATOR_BUILD_IDENTITY_INVALID",
                "evaluator dirty state is not a boolean",
            )
        if type(self.fixture_id) is not str or not self.fixture_id:
            raise EvaluatorScoringError(
                "EVALUATOR_BUILD_IDENTITY_INVALID",
                "evaluator fixture id is invalid",
            )
        if not _is_sha256(self.identity_sha256):
            raise EvaluatorScoringError(
                "EVALUATOR_BUILD_IDENTITY_INVALID",
                "evaluator build identity seal is invalid",
            )


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    raw_bytes: bytes
    file_sha256: str

    def validated_document(self) -> dict[str, Any]:
        observed_sha256 = hashlib.sha256(self.raw_bytes).hexdigest()
        if observed_sha256 != self.file_sha256:
            raise EvaluatorScoringError(
                "RESULT_SNAPSHOT_HASH_MISMATCH",
                "retained result bytes do not match their snapshot identity",
            )
        document = _decode_object_strict(
            self.raw_bytes, source="<retained-result-snapshot>", kind="simulator result"
        )
        return validate_result(document)


@dataclass(slots=True)
class _RowAccumulator:
    descriptor: MetricDescriptor
    kernel_family: str
    n_labeled: int = 0
    exact_count: int = 0
    zero_label_nonzero_prediction_count: int = 0
    absolute_errors: list[Fraction] = field(default_factory=list)
    relative_errors: list[Fraction] = field(default_factory=list)
    rate_errors_percentage_points: list[Fraction] = field(default_factory=list)
    unavailable: Counter[tuple[Any, ...]] = field(default_factory=Counter)
    noise_evaluated_count: int = 0
    noise_within_count: int = 0
    noise_not_applicable_count: int = 0
    noise_not_applicable_reasons: set[str] = field(default_factory=set)

    @property
    def n_scored(self) -> int:
        if self.descriptor.key in _COUNT_KEYS:
            return len(self.absolute_errors)
        if self.descriptor.key in _RATE_KEYS:
            return len(self.rate_errors_percentage_points)
        return 0

    @property
    def n_unavailable(self) -> int:
        return sum(self.unavailable.values())


@dataclass(frozen=True, slots=True)
class _PredictionValue:
    value: Fraction | None
    unavailable_key: tuple[Any, ...] | None


def load_synthetic_evaluator_build_identity(
    path: str | Path, *, fixture_id: str
) -> EvaluatorBuildIdentity:
    """Load one code-pinned, fixture-only evaluator build identity."""

    try:
        expected_file_sha256 = KNOWN_SYNTHETIC_BUILD_IDENTITY_FILE_SHA256[fixture_id]
    except KeyError as exc:
        raise authority_unavailable(
            AuthorityBlockerCode.BINDING_AUTHORITY_UNAVAILABLE,
            "synthetic evaluator build identity is not code-pinned",
        ) from exc

    material = _read_bounded(
        path, limit=_MAX_BUILD_IDENTITY_BYTES, kind="evaluator build identity"
    )
    observed_file_sha256 = hashlib.sha256(material).hexdigest()
    if observed_file_sha256 != expected_file_sha256:
        raise ContractValidationError(
            "EVALUATOR_BUILD_IDENTITY_FILE_HASH_MISMATCH",
            "evaluator build identity bytes do not match the code pin",
            context={
                "expected_sha256": expected_file_sha256,
                "observed_sha256": observed_file_sha256,
            },
        )
    document = _decode_object_strict(
        material, source=str(path), kind="evaluator build identity"
    )
    if frozenset(document) != _BUILD_IDENTITY_MEMBERS:
        raise ContractValidationError(
            "EVALUATOR_BUILD_IDENTITY_SCHEMA_INVALID",
            "evaluator build identity members are not exact",
        )
    if document["schema_version"] != _BUILD_IDENTITY_VERSION:
        raise ContractValidationError(
            "EVALUATOR_BUILD_IDENTITY_SCHEMA_INVALID",
            "evaluator build identity schema version is unsupported",
        )
    if document["authority_role"] != _SYNTHETIC_ROLE:
        raise ContractValidationError(
            "EVALUATOR_BUILD_IDENTITY_SCHEMA_INVALID",
            "evaluator build identity is not explicitly fixture-only",
        )
    if document["fixture_id"] != fixture_id:
        raise ContractValidationError(
            "EVALUATOR_BUILD_IDENTITY_PIN_MISMATCH",
            "evaluator build identity fixture id does not match the code pin",
        )
    if not _is_sha256(document.get("identity_sha256")):
        raise ContractValidationError(
            "EVALUATOR_BUILD_IDENTITY_SCHEMA_INVALID",
            "evaluator build identity seal is invalid",
        )
    expected_seal = canonical_sha256(document, "identity_sha256")
    if document["identity_sha256"] != expected_seal:
        raise ContractValidationError(
            "EVALUATOR_BUILD_IDENTITY_SEAL_MISMATCH",
            "evaluator build identity canonical seal does not match",
            context={
                "expected_sha256": expected_seal,
                "observed_sha256": document["identity_sha256"],
            },
        )
    return EvaluatorBuildIdentity(
        evaluator_git_commit=document["evaluator_git_commit"],
        evaluator_git_dirty=document["evaluator_git_dirty"],
        fixture_id=document["fixture_id"],
        identity_sha256=document["identity_sha256"],
    )


def load_result_snapshot(path: str | Path) -> ResultSnapshot:
    """Read, hash, strictly decode, and admit one bounded simulator result."""

    material = _read_bounded(path, limit=_MAX_RESULT_BYTES, kind="simulator result")
    document = _decode_object_strict(material, source=str(path), kind="simulator result")
    validate_result(document)
    return ResultSnapshot(
        raw_bytes=material,
        file_sha256=hashlib.sha256(material).hexdigest(),
    )


def score_synthetic_fixture(
    result_path: str | Path,
    labels_path: str | Path,
    label_authority_path: str | Path,
    evaluator_build_identity_path: str | Path,
    *,
    fixture_id: str,
) -> dict[str, Any]:
    """Score a code-pinned synthetic fixture without opening production labels."""

    labels = load_synthetic_matched_labels(
        labels_path,
        label_authority_path,
        authority_pin=known_synthetic_authority_pin(fixture_id),
    )
    evaluator_identity = load_synthetic_evaluator_build_identity(
        evaluator_build_identity_path, fixture_id=fixture_id
    )
    result = load_result_snapshot(result_path)
    return _build_accuracy_report(result, labels, evaluator_identity)


def _build_accuracy_report(
    result_snapshot: ResultSnapshot,
    labels: MatchedLabels,
    evaluator_identity: EvaluatorBuildIdentity,
) -> dict[str, Any]:
    """Build, seal, and self-validate one deterministic admitted report."""

    if labels.synthetic_fixture_id != evaluator_identity.fixture_id:
        raise EvaluatorScoringError(
            "SYNTHETIC_FIXTURE_IDENTITY_MISMATCH",
            "label and evaluator build fixtures do not share one identity",
        )

    registry = load_metric_registry()
    result = result_snapshot.validated_document()
    _verify_input_identities(result, labels, registry)

    prediction_launches = {
        canonical_launch_key_token(launch["launch_key"]): launch
        for launch in result["launches"]
    }
    label_launches = {
        launch.identity.canonical_token: launch for launch in labels.launches
    }
    families = tuple(sorted({launch.kernel_family for launch in labels.launches}))
    rows = {
        (descriptor.key, family): _RowAccumulator(descriptor, family)
        for descriptor in registry.metrics
        for family in families
    }

    comparison_failures: list[dict[str, Any]] = []
    total_cells = 0
    labeled_cells = 0

    for token in sorted(set(prediction_launches) | set(label_launches)):
        prediction_launch = prediction_launches.get(token)
        label_launch = label_launches.get(token)
        launch_key = (
            deepcopy(prediction_launch["launch_key"])
            if prediction_launch is not None
            else _launch_identity_model(label_launch.identity)  # type: ignore[union-attr]
        )
        for descriptor in registry.metrics:
            total_cells += 1
            label_metric = (
                label_launch.metric_by_key(descriptor.key)
                if label_launch is not None
                else None
            )
            prediction_metric = (
                prediction_launch["metrics"][descriptor.key]
                if prediction_launch is not None
                else None
            )
            row = (
                rows[(descriptor.key, label_launch.kernel_family)]
                if label_launch is not None
                else None
            )
            label_available = (
                label_metric is not None and label_metric.outcome_status == "available"
            )
            if label_available:
                labeled_cells += 1
                row.n_labeled += 1  # type: ignore[union-attr]

            if label_available and descriptor.key == CYCLE_METRIC_KEY:
                row.unavailable[_CYCLE_UNAVAILABLE_KEY] += 1  # type: ignore[union-attr]
                continue

            failure = _comparison_failure(
                launch_key=launch_key,
                descriptor=descriptor,
                label_launch=label_launch,
                label_metric=label_metric,
                prediction_metric=prediction_metric,
            )
            if failure is not None:
                comparison_failures.append(failure)
                continue

            prediction = _prediction_value(prediction_metric)
            if prediction.unavailable_key is not None:
                row.unavailable[prediction.unavailable_key] += 1  # type: ignore[union-attr]
                continue
            if prediction.value is None or label_metric is None or row is None:
                raise EvaluatorScoringError(
                    "SCORING_DISPOSITION_INCOMPLETE",
                    "candidate cell reached no terminal scoring disposition",
                    context={"metric_key": descriptor.key},
                )
            _score_numeric_cell(row, prediction.value, label_metric)

    per_metric_family = [
        _row_model(rows[(descriptor.key, family)])
        for descriptor in registry.metrics
        for family in families
    ]
    unavailable_counter: Counter[tuple[Any, ...]] = Counter()
    for row in rows.values():
        unavailable_counter.update(row.unavailable)
    scored_cells = sum(row.n_scored for row in rows.values())
    unavailable_cells = sum(unavailable_counter.values())
    comparison_failure_cells = len(comparison_failures)
    if scored_cells + unavailable_cells + comparison_failure_cells != total_cells:
        raise EvaluatorScoringError(
            "CANDIDATE_COVERAGE_INCOMPLETE",
            "candidate dispositions do not partition the full outer cell union",
        )

    report = {
        "schema_version": "accuracy_report.v1",
        "report_sha256": "0" * 64,
        "identities": _report_identities(
            result_snapshot, result, labels, evaluator_identity, registry
        ),
        "outcome": {
            "status": "admitted",
            "admission": {
                "status": "admitted",
                "checks": {
                    "result_schema": "passed",
                    "result_seal": "passed",
                    "registry_projection": "passed",
                    "architecture": "passed",
                    "matched_label_schema": "passed",
                    "matched_label_seal": "passed",
                    "crosswalk_identity": "passed",
                    "label_input_availability": "passed",
                    "units": "passed",
                },
                "failures": [],
            },
            "comparison_failures": comparison_failures,
            "coverage": {
                "total_cells": total_cells,
                "labeled_cells": labeled_cells,
                "unlabeled_cells": total_cells - labeled_cells,
                "scored_cells": scored_cells,
                "unavailable_cells": unavailable_cells,
                "comparison_failure_cells": comparison_failure_cells,
                "silent_absence_count": 0,
                "unavailable_by_type": _unavailable_models(
                    unavailable_counter, include_zero_cycles=True
                ),
            },
            "per_metric_family": per_metric_family,
        },
    }
    report["report_sha256"] = canonical_sha256(report, "report_sha256")
    validate_report(report, label_authority=labels.authority)
    return report


def _verify_input_identities(
    result: Mapping[str, Any], labels: MatchedLabels, registry: MetricRegistry
) -> None:
    card = result["arch_card_identity"]
    configuration = result["configuration_identity"]
    if configuration["arch_card_sha256"] != card["card_sha256"]:
        raise EvaluatorScoringError(
            "RESULT_IDENTITY_MISMATCH",
            "result configuration and architecture-card hashes disagree",
        )
    if (
        card["sm_arch"] != labels.binary_architecture
        or card["sku"] != labels.gpu_sku
    ):
        raise EvaluatorScoringError(
            "ARCHITECTURE_MISMATCH",
            "simulator result and matched labels describe different architectures",
            context={
                "result_architecture": card["sm_arch"],
                "result_sku": card["sku"],
                "label_architecture": labels.binary_architecture,
                "label_sku": labels.gpu_sku,
            },
        )
    for launch in labels.launches:
        if launch.identity.replay.binary_architecture != labels.binary_architecture:
            raise EvaluatorScoringError(
                "ARCHITECTURE_MISMATCH",
                "matched-label launch architecture contradicts its envelope",
            )
        for metric in launch.metrics:
            descriptor = registry.by_key(metric.key)
            if metric.unit != descriptor.unit:
                raise EvaluatorScoringError(
                    "UNIT_MISMATCH",
                    "matched-label metric unit contradicts the admitted projection",
                    context={"metric_key": metric.key},
                )
            if metric.ncu_metric != descriptor.ncu_metric:
                raise EvaluatorScoringError(
                    "METRIC_IDENTITY_MISMATCH",
                    "matched-label NCU metric contradicts the admitted projection",
                    context={"metric_key": metric.key},
                )


def _comparison_failure(
    *,
    launch_key: dict[str, Any],
    descriptor: MetricDescriptor,
    label_launch: MatchedLaunch | None,
    label_metric: LabelMetric | None,
    prediction_metric: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if label_launch is None:
        return {
            "failure_type": "MISSING_LABEL",
            "detail": "candidate cell has no exact matched-label launch",
            "context": {
                "launch_key": launch_key,
                "metric_key": descriptor.key,
            },
        }
    if prediction_metric is None:
        return {
            "failure_type": "MISSING_PREDICTION",
            "detail": "candidate cell has no exact simulator prediction launch",
            "context": {
                "launch_key": launch_key,
                "metric_key": descriptor.key,
                "kernel_family": label_launch.kernel_family,
            },
        }
    if label_metric is None:
        raise EvaluatorScoringError(
            "LABEL_METRIC_IDENTITY_MISSING",
            "validated matched-label launch lost an exact metric row",
        )
    if label_metric.unit != descriptor.unit:
        return {
            "failure_type": "UNIT_MISMATCH",
            "detail": "label unit does not match the retained metric projection",
            "context": {
                "launch_key": launch_key,
                "metric_key": descriptor.key,
                "kernel_family": label_launch.kernel_family,
                "expected_unit": descriptor.unit,
                "observed_unit": label_metric.unit,
            },
        }
    if label_metric.outcome_status != "available":
        return {
            "failure_type": "UNSUPPORTED_OUTCOME",
            "detail": "label outcome is not numeric evidence",
            "context": {
                "launch_key": launch_key,
                "metric_key": descriptor.key,
                "observed_status": f"label_{label_metric.outcome_status}",
            },
        }
    if prediction_metric.get("status") not in {"predicted", "unavailable"}:
        return {
            "failure_type": "UNSUPPORTED_OUTCOME",
            "detail": "simulator outcome is not supported by report v1",
            "context": {
                "launch_key": launch_key,
                "metric_key": descriptor.key,
                "observed_status": str(prediction_metric.get("status")),
            },
        }
    return None


def _prediction_value(metric: Mapping[str, Any]) -> _PredictionValue:
    if metric["status"] == "unavailable":
        detail = metric["detail"]
        return _PredictionValue(
            value=None,
            unavailable_key=(
                "unavailable",
                detail["reason"],
                detail["needs"],
                None,
                None,
            ),
        )
    value = metric["value"]
    if value["kind"] == "exact":
        return _PredictionValue(value=_as_fraction(value["value"]), unavailable_key=None)
    if value["denominator"] == 0:
        return _PredictionValue(
            value=None,
            unavailable_key=(
                "unavailable",
                "undefined_fraction",
                None,
                "zero_denominator",
                None,
            ),
        )
    return _PredictionValue(
        value=Fraction(value["numerator"], value["denominator"]),
        unavailable_key=None,
    )


def _score_numeric_cell(
    row: _RowAccumulator, prediction: Fraction, label: LabelMetric
) -> None:
    if label.value is None:
        raise EvaluatorScoringError(
            "LABEL_VALUE_MISSING",
            "available label outcome has no numeric value",
        )
    target = _as_fraction(label.value)
    native_error = prediction - target
    if row.descriptor.key in _COUNT_KEYS:
        row.absolute_errors.append(native_error)
        if target != 0:
            row.relative_errors.append(native_error / target)
        if target == 0 and prediction != 0:
            row.zero_label_nonzero_prediction_count += 1
    elif row.descriptor.key in _RATE_KEYS:
        row.rate_errors_percentage_points.append(native_error * 100)
    else:
        raise EvaluatorScoringError(
            "CYCLE_NUMERIC_SCORING_FORBIDDEN",
            "binding cycles reached the numeric scorer",
        )
    if prediction == target:
        row.exact_count += 1
    _score_noise(row, abs(native_error), target, label)


def _score_noise(
    row: _RowAccumulator,
    absolute_native_error: Fraction,
    target: Fraction,
    label: LabelMetric,
) -> None:
    dispersion = label.dispersion
    if dispersion.status != "available" or dispersion.scalar is None:
        row.noise_not_applicable_count += 1
        row.noise_not_applicable_reasons.add("dispersion_unavailable")
        return
    scalar = _as_fraction(dispersion.scalar)
    if dispersion.rule == "stddev_abs":
        threshold = scalar
    elif dispersion.rule in {"relative_tolerance", "coefficient_of_variation"}:
        threshold = abs(target) * scalar
    else:
        row.noise_not_applicable_count += 1
        row.noise_not_applicable_reasons.add("rule_not_absolute_interpretable")
        return
    row.noise_evaluated_count += 1
    if absolute_native_error <= threshold:
        row.noise_within_count += 1


def _row_model(row: _RowAccumulator) -> dict[str, Any]:
    common = {
        "metric_key": row.descriptor.key,
        "ncu_metric": row.descriptor.ncu_metric,
        "unit": row.descriptor.unit,
        "kernel_family": row.kernel_family,
        "n_labeled": row.n_labeled,
        "n_scored": row.n_scored,
        "n_unavailable": row.n_unavailable,
        "unavailable_by_type": (
            [
                {
                    "status": "required_unimplemented",
                    "blocking": CYCLE_BLOCKER,
                    "count": row.n_labeled,
                }
            ]
            if row.descriptor.key == CYCLE_METRIC_KEY
            else _unavailable_models(row.unavailable)
        ),
        "within_label_noise": {
            "comparison_semantics": (
                "absolute_error_native_unit_lte_declared_dispersion_threshold"
            ),
            "evaluated_count": row.noise_evaluated_count,
            "within_count": row.noise_within_count,
            "not_applicable_count": row.noise_not_applicable_count,
            "not_applicable_reasons": [
                reason
                for reason in _NOISE_REASON_ORDER
                if reason in row.noise_not_applicable_reasons
            ],
        },
    }
    if row.descriptor.key in _COUNT_KEYS:
        common["summary"] = {
            "kind": "count",
            "exact_count": row.exact_count,
            "zero_label_nonzero_prediction_count": (
                row.zero_label_nonzero_prediction_count
            ),
            "relative_error": _error_stats(row.relative_errors),
            "absolute_error": _error_stats(row.absolute_errors),
        }
    elif row.descriptor.key in _RATE_KEYS:
        common["summary"] = {
            "kind": "rate",
            "exact_count": row.exact_count,
            "error_percentage_points": _error_stats(
                row.rate_errors_percentage_points
            ),
        }
    else:
        common["summary"] = {
            "kind": "cycle",
            "status": "required_unimplemented",
            "blocking": CYCLE_BLOCKER,
        }
    return common


def _error_stats(errors: Sequence[Fraction]) -> dict[str, Any]:
    if not errors:
        return {
            "n": 0,
            "mean_absolute_error": None,
            "max_absolute_error": None,
            "signed_bias": None,
        }
    absolute = tuple(abs(error) for error in errors)
    return {
        "n": len(errors),
        "mean_absolute_error": _fraction_to_json(sum(absolute, Fraction()) / len(errors)),
        "max_absolute_error": _fraction_to_json(max(absolute)),
        "signed_bias": _fraction_to_json(sum(errors, Fraction()) / len(errors)),
    }


def _unavailable_models(
    counter: Mapping[tuple[Any, ...], int], *, include_zero_cycles: bool = False
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in _UNAVAILABLE_ORDER:
        count = int(counter.get(key, 0))
        if count == 0 and not (include_zero_cycles and key == _CYCLE_UNAVAILABLE_KEY):
            continue
        status, reason, needs, detail, blocking = key
        if status == "required_unimplemented":
            result.append({"status": status, "blocking": blocking, "count": count})
        elif reason == "undefined_fraction":
            result.append(
                {"status": status, "reason": reason, "detail": detail, "count": count}
            )
        else:
            result.append(
                {"status": status, "reason": reason, "needs": needs, "count": count}
            )
    unknown = {key for key, count in counter.items() if count and key not in _UNAVAILABLE_ORDER}
    if unknown:
        raise EvaluatorScoringError(
            "UNSUPPORTED_UNAVAILABLE_OUTCOME",
            "simulator emitted an unrecognized typed unavailability",
        )
    return result


def _report_identities(
    result_snapshot: ResultSnapshot,
    result: Mapping[str, Any],
    labels: MatchedLabels,
    evaluator_identity: EvaluatorBuildIdentity,
    registry: MetricRegistry,
) -> dict[str, Any]:
    simulator_dirty = result["build_identity"]["sim_git_dirty"]
    evaluator_dirty = evaluator_identity.evaluator_git_dirty
    return {
        "report_id": (
            f"synthetic:{evaluator_identity.fixture_id}:"
            f"{result_snapshot.file_sha256[:16]}:"
            f"{labels.matched_labels_sha256[:16]}"
        ),
        "result_schema_version": result["schema_version"],
        "result_file_sha256": result_snapshot.file_sha256,
        "result_canonical_sha256": result["result_sha256"],
        "registry_projection_id": "metric_registry_projection.v1",
        "registry_projection_sha256": registry.projection_sha256,
        "source_registry_sha256": registry.source_registry_sha256,
        "arch_card_id": result["arch_card_identity"]["card_id"],
        "arch_card_sha256": result["arch_card_identity"]["card_sha256"],
        "binary_architecture": result["arch_card_identity"]["sm_arch"],
        "gpu_sku": result["arch_card_identity"]["sku"],
        "simulator_git_dirty": simulator_dirty,
        "simulator_build_claimability": _claimability(simulator_dirty),
        "evaluator_git_commit": evaluator_identity.evaluator_git_commit,
        "evaluator_git_dirty": evaluator_dirty,
        "evaluator_build_claimability": _claimability(evaluator_dirty),
        "label_binding": {
            "status": "bound",
            "matched_labels_sha256": labels.matched_labels_sha256,
            "label_set_id": labels.label_set_id,
            "label_set_sha256": labels.label_set_sha256,
            "crosswalk_id": labels.crosswalk_id,
            "crosswalk_sha256": labels.crosswalk_sha256,
            "authority_role": "independently_admitted_label_and_crosswalk",
        },
    }


def _claimability(dirty: bool) -> dict[str, str]:
    if dirty:
        return {"status": "unclaimable", "reason": "dirty_build"}
    return {"status": "claimable", "basis": "clean_build"}


def _launch_identity_model(identity: LaunchIdentity) -> dict[str, Any]:
    return {
        "workload_id": identity.cohort.workload_id,
        "matched_execution_id": identity.cohort.matched_execution_id,
        "process_launch_ordinal": identity.cohort.process_launch_ordinal,
        "kernel_mangled_name": identity.replay.kernel_mangled_name,
        "grid": {
            "x": identity.replay.grid.x,
            "y": identity.replay.grid.y,
            "z": identity.replay.grid.z,
        },
        "block": {
            "x": identity.replay.block.x,
            "y": identity.replay.block.y,
            "z": identity.replay.block.z,
        },
        "device_id": identity.replay.device_id,
        "stream_id": identity.replay.stream_id,
        "binary_architecture": identity.replay.binary_architecture,
    }


def _as_fraction(value: int | float) -> Fraction:
    try:
        return Fraction(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluatorScoringError(
            "NUMERIC_SCORING_VALUE_INVALID",
            "numeric scoring value cannot be represented exactly",
        ) from exc


def _fraction_to_json(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    try:
        converted = float(value)
    except OverflowError as exc:
        raise EvaluatorScoringError(
            "NUMERIC_SCORING_RESULT_UNREPRESENTABLE",
            "numeric scoring statistic exceeds finite binary64",
        ) from exc
    if not math.isfinite(converted):
        raise EvaluatorScoringError(
            "NUMERIC_SCORING_RESULT_UNREPRESENTABLE",
            "numeric scoring statistic exceeds finite binary64",
        )
    return converted


def _read_bounded(path: str | Path, *, limit: int, kind: str) -> bytes:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            material = handle.read(limit + 1)
    except OSError as exc:
        raise ContractValidationError(
            "EVALUATOR_INPUT_READ_FAILED",
            f"could not read {kind}",
            context={"path": str(source), "error": str(exc)},
        ) from exc
    if len(material) > limit:
        raise ContractValidationError(
            "EVALUATOR_INPUT_TOO_LARGE",
            f"{kind} exceeds the bounded evaluator input size",
            context={"path": str(source), "max_bytes": limit},
        )
    return material


def _decode_object_strict(data: bytes, *, source: str, kind: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractValidationError(
            "EVALUATOR_INPUT_UTF8_INVALID",
            f"{kind} is not valid UTF-8",
            context={"source": source},
        ) from exc
    try:
        value = loads_json_strict(text, source=source)
    except ContractValidationError:
        raise
    except (MemoryError, OverflowError, RecursionError, ValueError) as exc:
        raise ContractValidationError(
            "EVALUATOR_INPUT_JSON_RESOURCE_INVALID",
            f"{kind} JSON exceeds deterministic parser limits",
            context={"source": source, "error_type": type(exc).__name__},
        ) from exc
    if type(value) is not dict:
        raise ContractValidationError(
            "EVALUATOR_INPUT_ROOT_INVALID",
            f"{kind} root must be an object",
            context={"source": source},
        )
    return value


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _canonical_json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _write_error(code: str, detail: str) -> None:
    sys.stderr.write(_canonical_json_line({"code": code, "detail": detail}))
    sys.stderr.flush()


def main(argv: Sequence[str] | None = None) -> int:
    args = tuple(sys.argv[1:] if argv is None else argv)
    if len(args) != 5:
        _write_error(
            "CLI_USAGE_FAILED",
            "expected: FIXTURE_ID RESULT LABELS LABEL_AUTHORITY EVALUATOR_IDENTITY",
        )
        return 2
    fixture_id, result_path, labels_path, label_authority_path, evaluator_path = args
    try:
        report = score_synthetic_fixture(
            result_path,
            labels_path,
            label_authority_path,
            evaluator_path,
            fixture_id=fixture_id,
        )
    except EvaluatorAuthorityUnavailable as exc:
        _write_error(exc.code.value, str(exc).partition(": ")[2])
        return 2
    except (ContractValidationError, EvaluatorScoringError) as exc:
        _write_error(exc.code, exc.detail)
        return 1
    sys.stdout.write(_canonical_json_line(report))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvaluatorBuildIdentity",
    "EvaluatorScoringError",
    "KNOWN_SYNTHETIC_BUILD_IDENTITY_FILE_SHA256",
    "ResultSnapshot",
    "load_result_snapshot",
    "load_synthetic_evaluator_build_identity",
    "main",
    "score_synthetic_fixture",
]
