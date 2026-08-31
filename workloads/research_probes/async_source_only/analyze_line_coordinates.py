#!/usr/bin/env python3
"""Analyze the fresh coordinate-corrected SM86 line-ownership campaign."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    reader = _SelectiveObjectReader(path.read_text(encoding="utf-8"), path)
    result = reader.object(fields)
    reader.whitespace()
    if reader.index != len(reader.text):
        raise base.AnalysisError(f"{path}: trailing JSON content")
    return result


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


def read_issue_labels(root: Path, orientation: str) -> dict[str, Any]:
    deltas = []
    for repetition in REPETITIONS:
        payload = read_direct_fields(
            direct_result_path(root, orientation, repetition), _ISSUE_FIELDS
        )
        expected_order = base.expected_order(repetition)
        if (
            payload.get("workload") != CONTRACT.workload
            or payload.get("mode") != "pair"
            or payload.get("groups") != CONTRACT.direct_groups
            or payload.get("order") != expected_order
            or payload.get("orientation") != orientation
            or payload.get("correct") is not True
        ):
            raise base.AnalysisError(f"{orientation} r{repetition}: direct identity mismatch")
        early = payload.get("early_fanout")
        late = payload.get("late_fanout")
        if not isinstance(early, dict) or not isinstance(late, dict):
            raise base.AnalysisError(f"{orientation} r{repetition}: missing arms")
        if any(
            arm.get("correct") is not True or arm.get("mismatches") != 0
            for arm in (early, late)
        ):
            raise base.AnalysisError(f"{orientation} r{repetition}: incorrect direct arm")
        derived = (
            base.integer(early.get("issue_cycles"), "early issue")
            - base.integer(late.get("issue_cycles"), "late issue")
        ) / CONTRACT.direct_groups
        retained = payload.get("paired_issue_delta_cycles_per_group")
        if not isinstance(retained, (int, float)) or not math.isclose(
            float(retained), derived, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise base.AnalysisError(f"{orientation} r{repetition}: issue delta mismatch")
        deltas.append(derived)
    return {"values": deltas, "median": statistics.median(deltas), "exact_zero": statistics.median(deltas) == 0.0}


def read_wait_labels(root: Path, orientation: str) -> dict[str, Any]:
    deltas = []
    for repetition in REPETITIONS:
        payload = read_direct_fields(
            direct_result_path(root, orientation, repetition), _WAIT_FIELDS
        )
        expected_order = base.expected_order(repetition)
        if (
            payload.get("workload") != CONTRACT.workload
            or payload.get("mode") != "pair"
            or payload.get("groups") != CONTRACT.direct_groups
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
                or arm.get("mismatches") != 0
                or arm.get("sum_consistent") is not True
                or base.integer(arm.get("issue_cycles"), f"{arm_name} issue")
                + base.integer(arm.get("wait_cycles"), f"{arm_name} wait")
                != base.integer(arm.get("total_cycles"), f"{arm_name} total")
            ):
                raise base.AnalysisError(
                    f"{orientation} r{repetition}: {arm_name} intervals do not sum"
                )
        derived = (
            base.integer(early.get("wait_cycles"), "early wait")
            - base.integer(late.get("wait_cycles"), "late wait")
        ) / CONTRACT.direct_groups
        retained = payload.get("paired_wait_delta_cycles_per_group")
        if not isinstance(retained, (int, float)) or not math.isclose(
            float(retained), derived, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise base.AnalysisError(f"{orientation} r{repetition}: wait delta mismatch")
        deltas.append(derived)
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
    return {key: report.get(key) for key in keys}


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
        return (
            CONTRACT.smoke_pass_status if smoke_passed else "complete_line_coordinates_structural_no_go",
            "Fresh coordinate smoke passed." if smoke_passed else "Fresh coordinate smoke failed a scientific gate.",
            0,
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
    contract.require_allowed_path(output)
    contract.require_allowed_path(output_md)
    contract.require_allowed_path(aggregate)
    if root != contract.evidence_root.resolve():
        raise base.AnalysisError(f"coordinate root {root} != frozen {contract.evidence_root}")
    if aggregate.resolve() != contract.aggregate_report.resolve():
        raise base.AnalysisError(
            f"coordinate aggregate {aggregate.resolve()} != frozen "
            f"{contract.aggregate_report.resolve()}"
        )
    authorization = execution.smoke_authorization_path(root)
    if smoke_only and publish:
        authorization.parent.mkdir(parents=True, exist_ok=True)
        authorization.write_text(
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
            + "\n"
        )
    errors: list[str] = []
    missing: list[str] = []
    bundles: dict[tuple[str, int], Any] = {}
    traces: dict[tuple[str, str], Any] = {}
    replays: dict[tuple[str, str], Any] = {}
    provenance: dict[tuple[str, int], Any] = {}
    binary = static = None
    label_reads = {
        "identification_issue_read": False,
        "identification_wait_read": False,
        "holdout_issue_read": False,
        "holdout_wait_read": False,
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
            issue = read_issue_labels(root, "first_line_early")
            label_reads["identification_issue_read"] = True
            if issue["exact_zero"]:
                wait = read_wait_labels(root, "first_line_early")
                label_reads["identification_wait_read"] = True
                identification = {
                    "issue": issue,
                    "wait": wait,
                    "signal": wait["material"],
                }
            else:
                identification = {"issue": issue, "wait": None, "signal": False}
            if identification["signal"]:
                holdout_issue = read_issue_labels(root, "second_line_early")
                label_reads["holdout_issue_read"] = True
                if holdout_issue["exact_zero"]:
                    holdout_wait = read_wait_labels(root, "second_line_early")
                    label_reads["holdout_wait_read"] = True
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
    status, decision, rc = outcome(errors=errors, missing=missing, smoke_only=smoke_only, smoke_passed=smoke_passed, identification=identification, holdout=holdout)
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
        "counts": {"ncu_bundles": len(bundles), "ncu_reports": 7 * len(bundles), "ncu_rows": 14 * len(bundles), "nvbit_cells": len(traces), "simulator_cells": len(replays), "direct_artifacts_validated": len(provenance)},
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
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_md_text = f"# Line-coordinate report\n\nStatus: `{status}`\n\n{decision}\n"
    aggregate_document = deterministic_aggregate(report)
    aggregate_text = json.dumps(aggregate_document, indent=2, sort_keys=True) + "\n"
    aggregate_written = False
    if publish:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        aggregate.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_text)
        output_md.write_text(report_md_text)
        aggregate_written = smoke_only or (
            rc == 0 and status.startswith("complete_line_coordinates_")
        )
        if aggregate_written:
            aggregate.write_text(aggregate_text)
        if smoke_only:
            authorization.write_text(aggregate_text)
    elif smoke_only and (
        not authorization.is_file()
        or authorization.read_text(encoding="utf-8") != aggregate_text
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
            if not path.is_file() or path.read_text(encoding="utf-8") != expected
        ]
        if mismatches:
            raise base.AnalysisError(
                f"retained full reports differ from current validation: {mismatches}"
            )
    print(json.dumps({"status": status, "output": str(output), "aggregate": str(aggregate), "aggregate_written": aggregate_written, "errors": len(errors), "missing": len(set(missing))}, sort_keys=True))
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
