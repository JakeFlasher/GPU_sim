#!/usr/bin/env python3
"""Analyze the fresh coordinate-corrected SM86 line-ownership campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from line_ownership_campaign_contract import (  # noqa: E402
    COORDINATE_CONTRACT,
    V1_CONTRACT,
    begin_campaign,
    configure_once,
    read_artifact_bytes,
    read_artifact_text,
    replace_artifact_text,
    validate_evidence_tree,
)

# Configuration is deliberately the first campaign action. Every imported
# producer/analyzer module binds this same immutable generation at import.
configure_once(COORDINATE_CONTRACT)

import analyze_line_ownership as base  # noqa: E402
import capture_line_ownership_execution as execution  # noqa: E402


CONTRACT = COORDINATE_CONTRACT
ORIENTATION_TO_CELL = {
    "first_line_early": execution.Cell("first"),
    "second_line_early": execution.Cell("second"),
}
REPETITIONS = tuple(range(1, 11))

_JSON_PRIMITIVE = re.compile(
    r"(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)
_ISSUE_FIELDS = {
    "workload": None,
    "mode": None,
    "groups": None,
    "order": None,
    "orientation": None,
    "correct": None,
    "early_fanout": {
        "issue_cycles": None,
        "correct": None,
        "mismatches": None,
    },
    "late_fanout": {
        "issue_cycles": None,
        "correct": None,
        "mismatches": None,
    },
    "paired_issue_delta_cycles_per_group": None,
}
_WAIT_FIELDS = {
    "workload": None,
    "mode": None,
    "groups": None,
    "order": None,
    "orientation": None,
    "correct": None,
    "early_fanout": {
        "issue_cycles": None,
        "wait_cycles": None,
        "total_cycles": None,
        "sum_consistent": None,
        "correct": None,
        "mismatches": None,
    },
    "late_fanout": {
        "issue_cycles": None,
        "wait_cycles": None,
        "total_cycles": None,
        "sum_consistent": None,
        "correct": None,
        "mismatches": None,
    },
    "paired_wait_delta_cycles_per_group": None,
}


class _SelectiveObjectReader:
    """Decode named fields while lexically skipping all other JSON values."""

    def __init__(self, text: str, context: Path) -> None:
        self.text = text
        self.context = context
        self.index = 0
        self.decoder = json.JSONDecoder()

    def whitespace(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def expect(self, token: str) -> None:
        self.whitespace()
        if not self.text.startswith(token, self.index):
            raise base.AnalysisError(
                f"{self.context}: expected {token!r} at byte {self.index}"
            )
        self.index += len(token)

    def decoded_value(self) -> Any:
        self.whitespace()
        try:
            value, end = self.decoder.raw_decode(self.text, self.index)
        except json.JSONDecodeError as error:
            raise base.AnalysisError(f"{self.context}: {error}") from error
        self.index = end
        return value

    def skip_value(self) -> None:
        self.whitespace()
        if self.index >= len(self.text):
            raise base.AnalysisError(f"{self.context}: truncated JSON value")
        first = self.text[self.index]
        if first == '"':
            self.decoded_value()
            return
        if first in "[{":
            stack = [first]
            self.index += 1
            while stack:
                if self.index >= len(self.text):
                    raise base.AnalysisError(
                        f"{self.context}: truncated JSON container"
                    )
                character = self.text[self.index]
                if character == '"':
                    self.decoded_value()
                    continue
                if character in "[{":
                    stack.append(character)
                elif character in "]}":
                    expected = "[" if character == "]" else "{"
                    if stack[-1] != expected:
                        raise base.AnalysisError(
                            f"{self.context}: mismatched JSON container"
                        )
                    stack.pop()
                self.index += 1
            return
        end = self.index
        while end < len(self.text) and self.text[end] not in ",]} \t\r\n":
            end += 1
        token = self.text[self.index:end]
        if not _JSON_PRIMITIVE.fullmatch(token):
            raise base.AnalysisError(
                f"{self.context}: malformed JSON primitive at byte {self.index}"
            )
        self.index = end

    def object(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        self.expect("{")
        result: dict[str, Any] = {}
        seen: set[str] = set()
        self.whitespace()
        if self.index < len(self.text) and self.text[self.index] == "}":
            self.index += 1
        else:
            while True:
                key = self.decoded_value()
                if not isinstance(key, str):
                    raise base.AnalysisError(f"{self.context}: non-string JSON key")
                if key in seen:
                    raise base.AnalysisError(
                        f"{self.context}: duplicate JSON key {key!r}"
                    )
                seen.add(key)
                self.expect(":")
                if key not in fields:
                    self.skip_value()
                elif isinstance(fields[key], Mapping):
                    result[key] = self.object(fields[key])
                else:
                    result[key] = self.decoded_value()
                self.whitespace()
                if self.index < len(self.text) and self.text[self.index] == ",":
                    self.index += 1
                    continue
                self.expect("}")
                break
        missing = sorted(set(fields) - set(result))
        if missing:
            raise base.AnalysisError(
                f"{self.context}: missing selected JSON fields {missing}"
            )
        return result


def read_direct_fields(path: Path, fields: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = read_artifact_text(CONTRACT, path)
    except (OSError, UnicodeError) as error:
        raise base.AnalysisError(f"cannot read {path}: {error}") from error
    return _select_direct_fields(text, path, fields)


def _select_direct_fields(
    text: str, context: Path, fields: Mapping[str, Any]
) -> dict[str, Any]:
    reader = _SelectiveObjectReader(text, context)
    result = reader.object(fields)
    reader.whitespace()
    if reader.index != len(reader.text):
        raise base.AnalysisError(f"{context}: trailing JSON content")
    return result


def read_admitted_direct_fields(
    path: Path,
    identity: Mapping[str, Any],
    fields: Mapping[str, Any],
    *,
    on_label_access: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Parse the same immutable result bytes admitted by provenance validation."""

    if not isinstance(identity, Mapping):
        raise base.AnalysisError(f"{path}: missing admitted result identity")
    retained_path = identity.get("path")
    if not isinstance(retained_path, str) or not Path(retained_path).is_absolute():
        raise base.AnalysisError(f"{path}: invalid admitted result path")
    admitted_path = CONTRACT.require_allowed_path(Path(retained_path))
    expected_path = CONTRACT.require_allowed_path(path)
    if admitted_path != expected_path:
        raise base.AnalysisError(
            f"{path}: admitted result path is {admitted_path}"
        )
    retained_size = identity.get("size_bytes")
    if isinstance(retained_size, bool) or not isinstance(retained_size, int):
        raise base.AnalysisError(f"{path}: invalid admitted result size")
    retained_sha256 = identity.get("sha256")
    if not isinstance(retained_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", retained_sha256
    ):
        raise base.AnalysisError(f"{path}: invalid admitted result SHA-256")
    try:
        content = read_artifact_bytes(CONTRACT, path)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise base.AnalysisError(f"cannot read {path}: {error}") from error
    if len(content) != retained_size:
        raise base.AnalysisError(
            f"{path}: result size changed after provenance admission"
        )
    if hashlib.sha256(content).hexdigest() != retained_sha256:
        raise base.AnalysisError(
            f"{path}: result bytes changed after provenance admission"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise base.AnalysisError(f"{path}: result is not UTF-8: {error}") from error
    if on_label_access is not None:
        on_label_access()
    return _select_direct_fields(text, path, fields)


def direct_counter(value: Any, context: str) -> int:
    """Require a finite, non-negative, integer-valued JSON number."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise base.AnalysisError(f"{context}: expected an integer counter, got {value!r}")
    if value < 0:
        raise base.AnalysisError(f"{context}: negative counter {value}")
    return value


def paired_delta(value: Any, context: str) -> float:
    """Require a finite numeric delta; booleans are not numeric evidence."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise base.AnalysisError(f"{context}: expected a finite numeric delta")
    try:
        result = float(value)
    except OverflowError as error:
        raise base.AnalysisError(f"{context}: expected a finite numeric delta") from error
    if not math.isfinite(result):
        raise base.AnalysisError(f"{context}: expected a finite numeric delta")
    return result


def new_label_access_state(expected_repetitions: int | None = None) -> dict[str, Any]:
    return {
        "state": "not_attempted",
        "attempted_repetitions": 0,
        "completed_repetitions": 0,
        "expected_repetitions": (
            len(REPETITIONS) if expected_repetitions is None else expected_repetitions
        ),
    }


def _begin_label_artifact(access: dict[str, Any]) -> None:
    access["attempted_repetitions"] += 1
    access["state"] = (
        "partial" if access["completed_repetitions"] else "attempted"
    )


def _complete_label_artifact(access: dict[str, Any]) -> None:
    access["completed_repetitions"] += 1
    access["state"] = (
        "complete"
        if access["completed_repetitions"] == access["expected_repetitions"]
        else "partial"
    )


def summarize_label_access(
    label_access: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        f"{name}_read": access.get("state") != "not_attempted"
        for name, access in label_access.items()
    }
    if any(
        access.get("state") in {"attempted", "partial"}
        for access in label_access.values()
    ):
        # The frozen valid report retains only its four boolean fields.  An
        # invalid read additionally says whether access merely started or
        # completed a strict subset of admitted repetitions.
        report["failure_access_state"] = {
            name: dict(access) for name, access in label_access.items()
        }
    return report


def validate_nvbit(
    root: Path, variant: str, orientation: str, binary: Mapping[str, Any]
) -> dict[str, Any]:
    cell = ORIENTATION_TO_CELL[orientation]
    destination = root / "nvbit" / cell.endpoint_name(variant)
    execution.validate_nvbit(root, destination, cell, variant, binary)
    value = base.nvbit_cell(root, variant, orientation, str(binary["sha256"]))
    # ``base.nvbit_cell`` has already parsed the protobuf and required exactly
    # one scalar preload per trace group, the frozen orientation mask, and the
    # complete per-lane address vector.  Reuse that validated oracle instead of
    # applying a second opcode/cache-policy heuristic to the same records.
    value["preload_ownership"] = {
        "records": value["ownership"]["preload_records"],
        "exact": value["ownership"]["preload_vector_exact"],
    }
    return value


def validate_replay(root: Path, variant: str, orientation: str) -> dict[str, Any]:
    cell = ORIENTATION_TO_CELL[orientation]
    destination = root / "simulator" / cell.endpoint_name(variant)
    execution.validate_replay(root, destination, cell, variant)
    return base.replay_cell(root, variant, orientation)


def validate_direct_provenance(
    root: Path, orientation: str, repetition: int, binary: Mapping[str, Any]
) -> dict[str, Any]:
    cell = ORIENTATION_TO_CELL[orientation]
    destination = root / "direct" / cell.direct_name / f"repeat-{repetition}"
    return execution.validate_direct_provenance(destination, cell, repetition, binary)


def direct_result_path(root: Path, orientation: str, repetition: int) -> Path:
    cell = ORIENTATION_TO_CELL[orientation]
    return root / "direct" / cell.direct_name / f"repeat-{repetition}/result.json"


def _admitted_result_identity(
    provenance: Mapping[tuple[str, int], Any], orientation: str, repetition: int
) -> Mapping[str, Any]:
    admitted = provenance.get((orientation, repetition))
    if not isinstance(admitted, Mapping) or not isinstance(
        admitted.get("result"), Mapping
    ):
        raise base.AnalysisError(
            f"{orientation} r{repetition}: no admitted direct-result identity"
        )
    return admitted["result"]


def read_issue_labels(
    root: Path,
    orientation: str,
    provenance: Mapping[tuple[str, int], Any],
    access: dict[str, Any],
) -> dict[str, Any]:
    deltas = []
    for repetition in REPETITIONS:
        path = direct_result_path(root, orientation, repetition)
        payload = read_admitted_direct_fields(
            path,
            _admitted_result_identity(provenance, orientation, repetition),
            _ISSUE_FIELDS,
            on_label_access=lambda: _begin_label_artifact(access),
        )
        expected_order = base.expected_order(repetition)
        groups = direct_counter(
            payload.get("groups"), f"{orientation} r{repetition}: groups"
        )
        if (
            payload.get("workload") != CONTRACT.workload
            or payload.get("mode") != "pair"
            or groups != CONTRACT.direct_groups
            or payload.get("order") != expected_order
            or payload.get("orientation") != orientation
            or payload.get("correct") is not True
        ):
            raise base.AnalysisError(f"{orientation} r{repetition}: direct identity mismatch")
        early = payload.get("early_fanout")
        late = payload.get("late_fanout")
        if not isinstance(early, dict) or not isinstance(late, dict):
            raise base.AnalysisError(f"{orientation} r{repetition}: missing arms")
        for arm_name, arm in (("early", early), ("late", late)):
            if arm.get("correct") is not True or direct_counter(
                arm.get("mismatches"),
                f"{orientation} r{repetition}: {arm_name} mismatches",
            ) != 0:
                raise base.AnalysisError(
                    f"{orientation} r{repetition}: incorrect direct arm"
                )
        derived = (
            direct_counter(
                early.get("issue_cycles"),
                f"{orientation} r{repetition}: early issue",
            )
            - direct_counter(
                late.get("issue_cycles"),
                f"{orientation} r{repetition}: late issue",
            )
        ) / CONTRACT.direct_groups
        retained = paired_delta(
            payload.get("paired_issue_delta_cycles_per_group"),
            f"{orientation} r{repetition}: paired issue delta",
        )
        if not math.isclose(
            retained, derived, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise base.AnalysisError(f"{orientation} r{repetition}: issue delta mismatch")
        deltas.append(derived)
        _complete_label_artifact(access)
    return {"values": deltas, "median": statistics.median(deltas), "exact_zero": statistics.median(deltas) == 0.0}


def read_wait_labels(
    root: Path,
    orientation: str,
    provenance: Mapping[tuple[str, int], Any],
    access: dict[str, Any],
) -> dict[str, Any]:
    deltas = []
    for repetition in REPETITIONS:
        path = direct_result_path(root, orientation, repetition)
        payload = read_admitted_direct_fields(
            path,
            _admitted_result_identity(provenance, orientation, repetition),
            _WAIT_FIELDS,
            on_label_access=lambda: _begin_label_artifact(access),
        )
        expected_order = base.expected_order(repetition)
        groups = direct_counter(
            payload.get("groups"), f"{orientation} r{repetition}: groups"
        )
        if (
            payload.get("workload") != CONTRACT.workload
            or payload.get("mode") != "pair"
            or groups != CONTRACT.direct_groups
            or payload.get("order") != expected_order
            or payload.get("orientation") != orientation
            or payload.get("correct") is not True
        ):
            raise base.AnalysisError(f"{orientation} r{repetition}: direct identity mismatch")
        early = payload["early_fanout"]
        late = payload["late_fanout"]
        for arm_name, arm in (("early", early), ("late", late)):
            if (
                arm.get("correct") is not True
                or direct_counter(
                    arm.get("mismatches"),
                    f"{orientation} r{repetition}: {arm_name} mismatches",
                )
                != 0
                or arm.get("sum_consistent") is not True
                or direct_counter(
                    arm.get("issue_cycles"),
                    f"{orientation} r{repetition}: {arm_name} issue",
                )
                + direct_counter(
                    arm.get("wait_cycles"),
                    f"{orientation} r{repetition}: {arm_name} wait",
                )
                != direct_counter(
                    arm.get("total_cycles"),
                    f"{orientation} r{repetition}: {arm_name} total",
                )
            ):
                raise base.AnalysisError(
                    f"{orientation} r{repetition}: {arm_name} intervals do not sum"
                )
        derived = (
            direct_counter(
                early.get("wait_cycles"),
                f"{orientation} r{repetition}: early wait",
            )
            - direct_counter(
                late.get("wait_cycles"),
                f"{orientation} r{repetition}: late wait",
            )
        ) / CONTRACT.direct_groups
        retained = paired_delta(
            payload.get("paired_wait_delta_cycles_per_group"),
            f"{orientation} r{repetition}: paired wait delta",
        )
        if not math.isclose(
            retained, derived, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise base.AnalysisError(f"{orientation} r{repetition}: wait delta mismatch")
        deltas.append(derived)
        _complete_label_artifact(access)
    return {
        "values": deltas,
        "median": statistics.median(deltas),
        "negative_repeats": sum(value < 0 for value in deltas),
        "material": statistics.median(deltas) <= -1.0
        and sum(value < 0 for value in deltas) >= 8,
    }


def deterministic_aggregate(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version", "generation_id", "campaign_contract", "assay", "mode", "status", "root", "binary",
        "decision", "counts", "smoke_gates", "criteria", "label_reads",
        "identification_gate", "holdout_gate", "schema18_prediction",
        "timing_promotion_authorized", "typed_visibility_tail_required",
        "missing_artifact_count", "validation_error_count",
    )
    result = {key: report.get(key) for key in keys}
    if report.get("mode") == "full":
        result["acceptance_criteria"] = report.get("acceptance_criteria")
    return result


def label_sequence_is_valid(
    identification: Mapping[str, Any] | None,
    holdout: Mapping[str, Any] | None,
    label_reads: Mapping[str, Any],
) -> bool:
    if identification is None or "failure_access_state" in label_reads:
        return False
    issue = identification.get("issue")
    if not isinstance(issue, Mapping):
        return False
    expected_identification_wait = issue.get("exact_zero") is True
    if label_reads.get("identification_issue_read") is not True or (
        label_reads.get("identification_wait_read")
        is not expected_identification_wait
    ):
        return False
    signal = identification.get("signal") is True
    if label_reads.get("holdout_issue_read") is not signal:
        return False
    if not signal:
        return holdout is None and label_reads.get("holdout_wait_read") is False
    if not isinstance(holdout, Mapping):
        return False
    holdout_issue = holdout.get("issue")
    if not isinstance(holdout_issue, Mapping):
        return False
    return label_reads.get("holdout_wait_read") is (
        holdout_issue.get("exact_zero") is True
    )


def full_acceptance_criteria(
    *,
    status: str,
    return_code: int,
    errors: Sequence[str],
    missing: Sequence[str],
    counts: Mapping[str, Any],
    smoke_gates: Mapping[str, Any] | None,
    identification: Mapping[str, Any] | None,
    holdout: Mapping[str, Any] | None,
    label_reads: Mapping[str, Any],
) -> dict[str, Any]:
    complete_statuses = {
        "complete_line_coordinates_identification_no_go",
        "complete_line_coordinates_holdout_no_go",
        "complete_line_coordinates_signal",
    }
    counts_exact = base.same_json(
        dict(counts),
        {
            "ncu_bundles": 20,
            "ncu_reports": 140,
            "ncu_rows": 280,
            "nvbit_cells": 4,
            "simulator_cells": 4,
            "direct_artifacts_validated": 20,
        },
    )
    structurally_valid = (
        not errors
        and not missing
        and isinstance(smoke_gates, Mapping)
        and bool(smoke_gates)
        and all(value is True for value in smoke_gates.values())
    )
    ac4_passed = (
        return_code == 0
        and status in complete_statuses
        and label_sequence_is_valid(identification, holdout, label_reads)
    )
    ac5_passed = (
        return_code == 0
        and status in complete_statuses
        and structurally_valid
        and counts_exact
    )
    return {
        "AC-4": {
            "passed": ac4_passed,
            "decision_status": status,
            "identification_precedes_holdout": ac4_passed,
            "label_reads": dict(label_reads),
        },
        "AC-5": {
            "passed": ac5_passed,
            "counts_exact": counts_exact,
            "missing_artifact_count": len(set(missing)),
            "validation_error_count": len(errors),
            "timing_unchanged": True,
            "timing_promotion_authorized": False,
            "typed_visibility_tail_required": True,
        },
    }


def outcome(
    *, errors: Sequence[str], missing: Sequence[str], smoke_only: bool,
    smoke_passed: bool, identification: Mapping[str, Any] | None,
    holdout: Mapping[str, Any] | None,
) -> tuple[str, str, int]:
    if errors:
        return "invalid_line_coordinates_campaign", "Malformed or identity-invalid evidence.", 1
    if missing:
        return "incomplete_line_coordinates_campaign", "Required evidence is incomplete.", 2
    if smoke_only:
        if smoke_passed:
            return CONTRACT.smoke_pass_status, "Fresh coordinate smoke passed.", 0
        return (
            "invalid_line_coordinates_campaign",
            "Fresh coordinate smoke failed a required structural gate.",
            1,
        )
    if identification is None:
        return "incomplete_line_coordinates_campaign", "Identification labels were not available.", 2
    if not identification.get("signal"):
        return "complete_line_coordinates_identification_no_go", "Identification did not show the frozen material signal.", 0
    if holdout is None:
        return "incomplete_line_coordinates_campaign", "Holdout labels were not evaluated.", 2
    if not holdout.get("signal"):
        return "complete_line_coordinates_holdout_no_go", "Identification passed but the untouched holdout did not.", 0
    return "complete_line_coordinates_signal", "Both frozen orientations show the material ownership signal.", 0


def smoke_pass_authorized(*, smoke_only: bool, status: str, return_code: int) -> bool:
    return (
        smoke_only
        and return_code == 0
        and status == CONTRACT.smoke_pass_status
    )


def validate_publication_paths(
    contract: Any,
    root: Path,
    output: Path,
    output_md: Path,
    aggregate: Path,
) -> tuple[Path, Path, Path]:
    """Freeze publications before any retained evidence is opened."""

    output = contract.require_allowed_path(output)
    output_md = contract.require_allowed_path(output_md)
    aggregate = contract.require_allowed_path(aggregate)
    expected_output = root / "analysis/report.json"
    expected_output_md = root / "analysis/report.md"
    expected_aggregate = contract.require_allowed_path(contract.aggregate_report)
    expected = (expected_output, expected_output_md, expected_aggregate)
    observed = (output, output_md, aggregate)
    if observed != expected:
        raise base.AnalysisError(
            f"coordinate publication paths {observed} != frozen {expected}"
        )
    if len(set(observed)) != len(observed):
        raise base.AnalysisError("coordinate publication paths must be distinct")
    protected = {
        root / "campaign_contract.json",
        execution.smoke_authorization_path(root),
    }
    aliases = sorted(str(path) for path in set(observed) & protected)
    if aliases:
        raise base.AnalysisError(
            f"coordinate publication aliases retained evidence: {aliases}"
        )
    for path in observed:
        cursor = path.parent
        while True:
            if cursor.is_symlink():
                raise base.AnalysisError(
                    f"coordinate publication ancestor is a symlink: {cursor}"
                )
            if cursor == cursor.parent:
                break
            cursor = cursor.parent
        if path.is_symlink():
            raise base.AnalysisError(
                f"coordinate publication is a symlink: {path}"
            )
        if path.exists():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise base.AnalysisError(
                    f"coordinate publication is not one regular file: {path}"
                )
    return observed


def run(
    root: Path,
    output: Path,
    output_md: Path,
    aggregate: Path,
    *,
    smoke_only: bool,
    publish: bool = True,
) -> int:
    contract = begin_campaign()
    root = contract.require_allowed_path(root)
    if root != contract.require_allowed_path(contract.evidence_root):
        raise base.AnalysisError(f"coordinate root {root} != frozen {contract.evidence_root}")
    output, output_md, aggregate = validate_publication_paths(
        contract, root, output, output_md, aggregate
    )
    validate_evidence_tree(contract, root)
    authorization = execution.smoke_authorization_path(root)
    if smoke_only and publish:
        replace_artifact_text(
            contract,
            authorization,
            json.dumps(
                {
                    "schema_version": 1,
                    "generation_id": contract.generation_id,
                    "root": str(root),
                    "status": "smoke_validation_in_progress",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    errors: list[str] = []
    missing: list[str] = []
    bundles: dict[tuple[str, int], Any] = {}
    traces: dict[tuple[str, str], Any] = {}
    replays: dict[tuple[str, str], Any] = {}
    provenance: dict[tuple[str, int], Any] = {}
    binary = static = None
    label_access = {
        "identification_issue": new_label_access_state(),
        "identification_wait": new_label_access_state(),
        "holdout_issue": new_label_access_state(),
        "holdout_wait": new_label_access_state(),
    }
    smoke_gates = None
    identification = holdout = None
    try:
        execution.validate_campaign_contract_binding(root)
        binary = execution.retained_binary(root)
        execution.support.validate_static(root)
        static = base.static_report(root)
        if not static["artifact_valid"]:
            errors.extend(static["artifact_errors"])
    except FileNotFoundError as error:
        missing.append(str(error))
    except Exception as error:
        errors.append(str(error))
    if not errors and not missing and binary:
        try:
            bundles[("first_line_early", 1)] = base.validate_bundle(root, "first_line_early", 1, binary)
            for variant in contract.variants:
                traces[(variant, "first_line_early")] = validate_nvbit(root, variant, "first_line_early", binary)
            base.require_cross_endpoint_tracer_identity(list(traces.values()))
            for variant in contract.variants:
                replays[(variant, "first_line_early")] = validate_replay(root, variant, "first_line_early")
            provenance[("first_line_early", 1)] = validate_direct_provenance(root, "first_line_early", 1, binary)
            early, late = replays[("early_fanout", "first_line_early")], replays[("late_fanout", "first_line_early")]
            smoke_gates = {
                "static_assay_valid": static["assay_valid"],
                "ncu_contract_exact": bundles[("first_line_early", 1)]["valid"],
                "per_lane_ownership_exact": all(value["ownership"]["per_lane_ownership_exact"] and value["preload_ownership"]["exact"] for value in traces.values()),
                "aggregate_coordinates_exact": all(all(value["aggregate_gates"].values()) for value in replays.values()),
                "schema18_paired_tie": early["schema18_complete"] == late["schema18_complete"],
                "production_paired_tie": early["production"] == late["production"],
                "direct_provenance_valid": True,
            }
        except FileNotFoundError as error:
            missing.append(str(error))
        except Exception as error:
            errors.append(str(error))
    smoke_passed = bool(smoke_gates) and all(smoke_gates.values())
    if smoke_passed and not smoke_only and binary:
        try:
            execution.require_full_campaign_authorization(root, binary)
        except FileNotFoundError as error:
            missing.append(str(error))
        except Exception as error:
            errors.append(str(error))
        if not errors and not missing:
            for orientation in contract.orientations:
                for repetition in REPETITIONS:
                    try:
                        bundles[(orientation, repetition)] = base.validate_bundle(root, orientation, repetition, binary)
                    except FileNotFoundError as error:
                        missing.append(str(error))
                    except Exception as error:
                        errors.append(str(error))
        if not errors and not missing:
            for orientation in contract.orientations:
                for variant in contract.variants:
                    key = (variant, orientation)
                    try:
                        traces[key] = validate_nvbit(root, variant, orientation, binary)
                    except FileNotFoundError as error:
                        missing.append(str(error))
                    except Exception as error:
                        errors.append(str(error))
        if not errors and not missing:
            try:
                base.require_cross_endpoint_tracer_identity(list(traces.values()))
            except Exception as error:
                errors.append(str(error))
        if not errors and not missing:
            for orientation in contract.orientations:
                for variant in contract.variants:
                    key = (variant, orientation)
                    try:
                        replays[key] = validate_replay(root, variant, orientation)
                    except FileNotFoundError as error:
                        missing.append(str(error))
                    except Exception as error:
                        errors.append(str(error))
        if not errors and not missing:
            try:
                if not all(
                    all(value["aggregate_gates"].values())
                    for value in replays.values()
                ):
                    raise base.AnalysisError(
                        "one or more replay coordinate gates failed"
                    )
                for orientation in contract.orientations:
                    early = replays[("early_fanout", orientation)]
                    late = replays[("late_fanout", orientation)]
                    if early["schema18_complete"] != late["schema18_complete"]:
                        raise base.AnalysisError(
                            f"{orientation}: complete schema18 observation did not tie"
                        )
                    if early["production"] != late["production"]:
                        raise base.AnalysisError(
                            f"{orientation}: production fingerprint did not tie"
                        )
                if not all(value["valid"] for value in bundles.values()):
                    raise base.AnalysisError("one or more coordinate NCU bundles failed")
            except Exception as error:
                errors.append(str(error))
        if not errors and not missing:
            for orientation in contract.orientations:
                for repetition in REPETITIONS:
                    try:
                        provenance[(orientation, repetition)] = (
                            validate_direct_provenance(
                                root, orientation, repetition, binary
                            )
                        )
                    except FileNotFoundError as error:
                        missing.append(str(error))
                    except Exception as error:
                        errors.append(str(error))
    if smoke_passed and not smoke_only and not errors and not missing and len(bundles) == 20:
        try:
            issue = read_issue_labels(
                root,
                "first_line_early",
                provenance,
                label_access["identification_issue"],
            )
            if issue["exact_zero"]:
                wait = read_wait_labels(
                    root,
                    "first_line_early",
                    provenance,
                    label_access["identification_wait"],
                )
                identification = {
                    "issue": issue,
                    "wait": wait,
                    "signal": wait["material"],
                }
            else:
                identification = {"issue": issue, "wait": None, "signal": False}
            if identification["signal"]:
                holdout_issue = read_issue_labels(
                    root,
                    "second_line_early",
                    provenance,
                    label_access["holdout_issue"],
                )
                if holdout_issue["exact_zero"]:
                    holdout_wait = read_wait_labels(
                        root,
                        "second_line_early",
                        provenance,
                        label_access["holdout_wait"],
                    )
                    holdout = {
                        "issue": holdout_issue,
                        "wait": holdout_wait,
                        "signal": holdout_wait["material"],
                    }
                else:
                    holdout = {
                        "issue": holdout_issue,
                        "wait": None,
                        "signal": False,
                    }
        except FileNotFoundError as error:
            missing.append(str(error))
        except Exception as error:
            errors.append(str(error))
    label_reads = summarize_label_access(label_access)
    status, decision, rc = outcome(errors=errors, missing=missing, smoke_only=smoke_only, smoke_passed=smoke_passed, identification=identification, holdout=holdout)
    counts = {
        "ncu_bundles": len(bundles),
        "ncu_reports": 7 * len(bundles),
        "ncu_rows": 14 * len(bundles),
        "nvbit_cells": len(traces),
        "simulator_cells": len(replays),
        "direct_artifacts_validated": len(provenance),
    }
    report = {
        "schema_version": 1,
        "generation_id": contract.generation_id,
        "campaign_contract": contract.to_json(),
        "assay": contract.assay_id,
        "mode": "smoke_only" if smoke_only else "full",
        "status": status,
        "root": str(root),
        "binary": binary,
        "decision": decision,
        "counts": counts,
        "static": static,
        "smoke_gates": smoke_gates,
        "identification_gate": identification,
        "holdout_gate": holdout,
        "label_reads": label_reads,
        "schema18_prediction": "exact paired tie / zero",
        "criteria": {"static_passed": None if static is None else static["assay_valid"], "smoke_passed": smoke_passed, "identification_signal": None if identification is None else identification["signal"], "holdout_signal": None if holdout is None else holdout["signal"], "schema18_tie": None if smoke_gates is None else smoke_gates["schema18_paired_tie"], "timing_unchanged": True},
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "missing_artifact_count": len(set(missing)),
        "validation_error_count": len(errors),
        "missing_artifacts": sorted(set(missing)),
        "validation_errors": errors,
    }
    if not smoke_only:
        report["acceptance_criteria"] = full_acceptance_criteria(
            status=status,
            return_code=rc,
            errors=errors,
            missing=missing,
            counts=counts,
            smoke_gates=smoke_gates,
            identification=identification,
            holdout=holdout,
            label_reads=label_reads,
        )
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_md_text = f"# Line-coordinate report\n\nStatus: `{status}`\n\n{decision}\n"
    aggregate_document = deterministic_aggregate(report)
    aggregate_text = json.dumps(aggregate_document, indent=2, sort_keys=True) + "\n"
    aggregate_written = False
    pass_authorized = smoke_pass_authorized(
        smoke_only=smoke_only, status=status, return_code=rc
    )
    validate_evidence_tree(contract, root)
    if publish:
        replace_artifact_text(contract, output, report_text)
        replace_artifact_text(contract, output_md, report_md_text)
        aggregate_written = pass_authorized or (
            rc == 0 and status.startswith("complete_line_coordinates_")
        )
        if aggregate_written:
            replace_artifact_text(contract, aggregate, aggregate_text)
        if smoke_only:
            # A failed smoke replaces any stale pass token with an explicitly
            # non-authorizing record, but never publishes the global aggregate.
            replace_artifact_text(contract, authorization, aggregate_text)
    elif smoke_only and (
        not authorization.is_file()
        or read_artifact_text(contract, authorization) != aggregate_text
    ):
        raise base.AnalysisError(
            "retained smoke authorization differs from current smoke validation"
        )
    elif not smoke_only:
        retained = (
            (output, report_text),
            (output_md, report_md_text),
            (aggregate, aggregate_text),
        )
        mismatches = [
            str(path)
            for path, expected in retained
            if not path.is_file()
            or read_artifact_text(contract, path) != expected
        ]
        if mismatches:
            raise base.AnalysisError(
                f"retained full reports differ from current validation: {mismatches}"
            )
    validate_evidence_tree(contract, root)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "aggregate": str(aggregate),
                "aggregate_written": aggregate_written,
                "errors": len(errors),
                "missing": len(set(missing)),
                "counts": report["counts"],
                "label_reads": label_reads,
                "identification_signal": (
                    None if identification is None else identification["signal"]
                ),
                "holdout_signal": None if holdout is None else holdout["signal"],
                "timing_unchanged": True,
                "timing_promotion_authorized": False,
                "acceptance_criteria": report.get("acceptance_criteria"),
            },
            sort_keys=True,
        )
    )
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CONTRACT.evidence_root)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--aggregate-output", type=Path, default=CONTRACT.aggregate_report)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate without changing reports; smoke also checks retained authorization",
    )
    args = parser.parse_args(argv)
    output = args.output or args.root / "analysis/report.json"
    output_md = args.output_md or output.with_suffix(".md")
    return run(
        args.root,
        output,
        output_md,
        args.aggregate_output,
        smoke_only=args.smoke_only,
        publish=not args.validate_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
