#!/usr/bin/env python3
"""Build the current, row-granular SOL native/proxy coverage ledger.

The historical v1 inventory explains the frozen 344-cell authority.  This
artifact answers a different question: for every current SOL workload row,
what did the fresh live probe prove, which exact supplemental matrix rows are
scheduled, and why is any remaining row unscheduled?  Reduced proxies are kept
structurally separate from native evidence and architecture-inapplicable rows.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_sol_workload_capture_matrix as matrix_builder  # noqa: E402
import sol_reconstitution as runnability_tool  # noqa: E402
import sol_shrink_apply as shrink_tool  # noqa: E402
import sol_workload_coverage_inventory as historical  # noqa: E402


SCHEMA_PATH = Path(
    "workloads/simbench_bw/schemas/"
    "sol_live_workload_coverage_inventory.schema.json"
)
EXPECTED_PROBLEMS = 235
EXPECTED_ROWS = 3957
EXPECTED_ARCHITECTURE_PROBLEMS = 34
EXPECTED_ARCHITECTURE_ROWS = 537
EXPECTED_IN_SCOPE_PROBLEMS = 201
EXPECTED_IN_SCOPE_ROWS = 3420
COHORT_ORDER = {"native": 0, "partial_native_row": 1, "reduced_proxy": 2}
SCHEDULE_STATES = (
    "architecture_inapplicable",
    "in_scope_unscheduled",
    "native_scheduled",
    "partial_native_scheduled",
    "reduced_proxy_scheduled",
    "partial_native_and_reduced_proxy_scheduled",
)
PLAN_OWNER_FILENAME = ".sol-live-coverage-plan-owner-v1.json"
PLAN_OWNER_KIND = "sol_live_coverage_plan_owner_v1"


class LiveCoverageError(ValueError):
    """Current evidence and scheduled matrix rows do not reconcile."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveCoverageError(message)


def _diagnostics_by_index(
    *, slug: str, source_rows: list[dict[str, Any]], run_cell: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    diagnostics = run_cell.get("workload_diagnostics")
    _require(
        isinstance(diagnostics, list) and len(diagnostics) == len(source_rows),
        f"live runnability diagnostic cardinality mismatch: {slug}",
    )
    by_index: dict[int, dict[str, Any]] = {}
    for expected_index, (source, diagnostic) in enumerate(
        zip(source_rows, diagnostics)
    ):
        _require(
            isinstance(diagnostic, dict)
            and diagnostic.get("workload_index") == expected_index
            and diagnostic.get("source_line") == source["source_line"]
            and diagnostic.get("uuid") == source["source_workload_uuid"]
            and diagnostic.get("axes") == source["source_axes"]
            and diagnostic.get("workload_sha256")
            == source["source_row_sha256"],
            f"live runnability diagnostic/source mismatch: {slug}:{expected_index}",
        )
        by_index[expected_index] = diagnostic
    return by_index


def _shrink_indexes(
    shrink_document: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
    cells = shrink_document.get("cells")
    _require(isinstance(cells, list), "live shrink cells are absent")
    by_slug: dict[str, dict[str, Any]] = {}
    plans: dict[tuple[str, int, str], dict[str, Any]] = {}
    for cell in cells:
        _require(isinstance(cell, dict), "live shrink cell is malformed")
        slug = cell.get("slug")
        _require(
            isinstance(slug, str) and slug and slug not in by_slug,
            f"live shrink slug is duplicated/invalid: {slug!r}",
        )
        by_slug[slug] = cell
        row_plans = cell.get("row_plans")
        _require(isinstance(row_plans, list), f"live shrink plans absent: {slug}")
        for plan in row_plans:
            _require(isinstance(plan, dict), f"live shrink plan malformed: {slug}")
            key = (
                slug,
                plan.get("source_workload_index"),
                plan.get("source_workload_uuid"),
            )
            _require(
                type(key[1]) is int and isinstance(key[2], str)
                and key not in plans,
                f"live shrink plan identity duplicated/invalid: {key!r}",
            )
            plans[key] = plan
    return by_slug, plans


def _schedule_index(
    *, matrix: dict[str, Any], problems: dict[str, dict[str, Any]],
    run_cells: dict[str, dict[str, Any]],
    shrink_plans: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    matrix_rows = matrix.get("rows")
    _require(isinstance(matrix_rows, list), "live supplemental matrix rows are absent")
    result: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[str, str, str]] = set()
    for row in matrix_rows:
        _require(isinstance(row, dict), "live supplemental matrix row is malformed")
        key = row.get("key")
        _require(isinstance(key, dict), "live supplemental matrix key is malformed")
        slug = key.get("slug")
        uuid = key.get("source_workload_uuid")
        variant_id = key.get("variant_id")
        identity = (slug, uuid, variant_id)
        _require(
            all(isinstance(item, str) and item for item in identity)
            and identity not in seen_keys,
            f"live supplemental matrix key is duplicated/invalid: {identity!r}",
        )
        seen_keys.add(identity)
        _require(slug in problems and slug in run_cells,
                 f"matrix schedules an out-of-scope SOL problem: {slug!r}")
        source_rows = problems[slug]["rows"]
        source_index = row.get("source_workload_index")
        _require(
            type(source_index) is int and 0 <= source_index < len(source_rows),
            f"matrix source row index is invalid: {slug}:{source_index!r}",
        )
        source = source_rows[source_index]
        _require(
            uuid == source["source_workload_uuid"]
            and row.get("source_workload_uuid") == uuid
            and row.get("source_axes") == source["source_axes"]
            and row.get("source_row_sha256") == source["source_row_sha256"],
            f"matrix source row binding mismatch: {slug}:{source_index}",
        )
        cohort = row.get("cohort")
        _require(cohort in COHORT_ORDER,
                 f"matrix row cohort is invalid: {slug}:{cohort!r}")
        diagnostic = run_cells[slug]["_diagnostics"][source_index]
        passed = (
            diagnostic.get("diagnostic_present") is True
            and str(diagnostic.get("status")).lower() == "passed"
        )
        if cohort == "native":
            _require(
                run_cells[slug].get("status") == "runnable" and passed
                and variant_id == "native"
                and row.get("problem_level_runnable") is True
                and row.get("native_shape") is True
                and row.get("native_predictive") is True,
                f"native matrix row lacks full live evidence: {slug}:{source_index}",
            )
        elif cohort == "partial_native_row":
            _require(
                run_cells[slug].get("status") != "runnable" and passed
                and variant_id != "native"
                and row.get("problem_level_runnable") is False
                and row.get("native_shape") is True
                and row.get("native_predictive") is False,
                f"partial-native row lacks exact passed diagnostic: "
                f"{slug}:{source_index}",
            )
        else:
            plan = shrink_plans.get((slug, source_index, uuid))
            _require(
                isinstance(plan, dict)
                and plan.get("planning_status") == "proxy_planned"
                and plan.get("correctness_result") == "passed"
                and plan.get("reduced_run_status") == "runnable"
                and plan.get("variant_id") == variant_id
                and row.get("problem_level_runnable") is False
                and row.get("native_shape") is False
                and row.get("native_predictive") is False,
                f"reduced-proxy row lacks exact passed plan: {slug}:{source_index}",
            )
        result[(slug, source_index)].append(copy.deepcopy(row))
    for rows in result.values():
        rows.sort(key=lambda row: (
            COHORT_ORDER[row["cohort"]], row["key"]["variant_id"]
        ))
        cohorts = [row["cohort"] for row in rows]
        _require(
            len(cohorts) == len(set(cohorts)),
            "one source workload row has duplicate schedule cohorts",
        )
        _require(
            not ("native" in cohorts and len(cohorts) != 1),
            "full-native rows cannot also carry partial/proxy schedules",
        )
    return result


def _reason_document(
    *, architecture_reason: str | None, run_cell: dict[str, Any] | None,
    diagnostic: dict[str, Any] | None, shrink_cell: dict[str, Any] | None,
    shrink_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if architecture_reason is not None:
        reason = {
            "code": "architecture_inapplicable",
            "architecture_reason": architecture_reason,
            "runnability_status": None,
            "diagnostic_present": None,
            "diagnostic_status": None,
            "shrink_classification": None,
            "shrink_planning_status": None,
            "shrink_correctness_result": None,
            "shrink_reduced_run_status": None,
            "shrink_planning_reason": None,
        }
    else:
        reason = {
            "code": "no_admitted_capture_variant",
            "architecture_reason": None,
            "runnability_status": (
                run_cell.get("status") if isinstance(run_cell, dict) else None
            ),
            "diagnostic_present": (
                diagnostic.get("diagnostic_present")
                if isinstance(diagnostic, dict) else None
            ),
            "diagnostic_status": (
                diagnostic.get("status") if isinstance(diagnostic, dict) else None
            ),
            "shrink_classification": (
                shrink_cell.get("classification")
                if isinstance(shrink_cell, dict) else None
            ),
            "shrink_planning_status": (
                shrink_plan.get("planning_status")
                if isinstance(shrink_plan, dict) else None
            ),
            "shrink_correctness_result": (
                shrink_plan.get("correctness_result")
                if isinstance(shrink_plan, dict) else None
            ),
            "shrink_reduced_run_status": (
                shrink_plan.get("reduced_run_status")
                if isinstance(shrink_plan, dict) else None
            ),
            "shrink_planning_reason": (
                shrink_plan.get("planning_reason")
                if isinstance(shrink_plan, dict) else None
            ),
        }
    return {**reason, "reason_evidence_sha256": canonical_sha256(reason)}


def _schedule_state(cohorts: list[str], *, architecture: bool) -> str:
    if architecture:
        _require(not cohorts, "architecture-inapplicable row was scheduled")
        return "architecture_inapplicable"
    if cohorts == []:
        return "in_scope_unscheduled"
    if cohorts == ["native"]:
        return "native_scheduled"
    if cohorts == ["partial_native_row"]:
        return "partial_native_scheduled"
    if cohorts == ["reduced_proxy"]:
        return "reduced_proxy_scheduled"
    if cohorts == ["partial_native_row", "reduced_proxy"]:
        return "partial_native_and_reduced_proxy_scheduled"
    raise LiveCoverageError(f"unsupported source-row schedule combination: {cohorts!r}")


def build_row_ledger(
    *, problems: dict[str, dict[str, Any]],
    included: dict[str, dict[str, Any]],
    excluded_scope: dict[str, dict[str, Any]],
    runnability: dict[str, Any], shrink_document: dict[str, Any],
    matrix: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Reconcile every source row; inputs must already pass producer gates."""
    _require(
        set(problems) == set(included) | set(excluded_scope)
        and not (set(included) & set(excluded_scope)),
        "port scope does not partition the current SOL corpus",
    )
    cells = runnability.get("cells")
    _require(isinstance(cells, list), "live runnability cells are absent")
    run_cells: dict[str, dict[str, Any]] = {}
    for cell in cells:
        _require(isinstance(cell, dict), "live runnability cell is malformed")
        slug = cell.get("slug")
        _require(
            isinstance(slug, str) and slug in included and slug not in run_cells,
            f"live runnability slug is duplicated/out-of-scope: {slug!r}",
        )
        source_rows = problems[slug]["rows"]
        bound = copy.deepcopy(cell)
        bound["_diagnostics"] = _diagnostics_by_index(
            slug=slug, source_rows=source_rows, run_cell=cell,
        )
        run_cells[slug] = bound
    _require(set(run_cells) == set(included),
             "live runnability does not cover the exact 201-problem scope")

    shrink_cells, shrink_plans = _shrink_indexes(shrink_document)
    _require(set(shrink_cells) <= set(included),
             "live shrink authority contains an out-of-scope problem")
    schedules = _schedule_index(
        matrix=matrix, problems=problems, run_cells=run_cells,
        shrink_plans=shrink_plans,
    )

    row_records: list[dict[str, Any]] = []
    problem_records: list[dict[str, Any]] = []
    schedule_counter: Counter[str] = Counter()
    accounted_matrix_rows = 0
    for slug in sorted(problems, key=lambda item: (problems[item]["category"], item)):
        source = problems[slug]
        architecture = slug in excluded_scope
        run_cell = run_cells.get(slug)
        shrink_cell = shrink_cells.get(slug)
        problem_rows: list[dict[str, Any]] = []
        for native_row in source["rows"]:
            index = native_row["source_workload_index"]
            uuid = native_row["source_workload_uuid"]
            selected = schedules.get((slug, index), [])
            cohorts = [row["cohort"] for row in selected]
            state = _schedule_state(cohorts, architecture=architecture)
            diagnostic = (
                run_cell["_diagnostics"][index]
                if isinstance(run_cell, dict) else None
            )
            shrink_plan = shrink_plans.get((slug, index, uuid))
            if not architecture:
                passed = (
                    isinstance(diagnostic, dict)
                    and diagnostic.get("diagnostic_present") is True
                    and str(diagnostic.get("status")).lower() == "passed"
                )
                _require(
                    passed == any(
                        cohort in {"native", "partial_native_row"}
                        for cohort in cohorts
                    ),
                    f"native passed-row scheduling mismatch: {slug}:{index}",
                )
                proxy_admitted = (
                    isinstance(shrink_plan, dict)
                    and shrink_plan.get("planning_status") == "proxy_planned"
                    and shrink_plan.get("correctness_result") == "passed"
                    and shrink_plan.get("reduced_run_status") == "runnable"
                )
                _require(
                    proxy_admitted == ("reduced_proxy" in cohorts),
                    f"reduced-proxy scheduling mismatch: {slug}:{index}",
                )
            reason = None
            if not selected:
                reason = _reason_document(
                    architecture_reason=(
                        excluded_scope[slug]["reason"] if architecture else None
                    ),
                    run_cell=run_cell, diagnostic=diagnostic,
                    shrink_cell=shrink_cell, shrink_plan=shrink_plan,
                )
            row_record = {
                "category": source["category"],
                "slug": slug,
                "source_workload_index": index,
                "source_workload_uuid": uuid,
                "source_axes": copy.deepcopy(native_row["source_axes"]),
                "source_row_sha256": native_row["source_row_sha256"],
                "schedule_state": state,
                "schedule_cohorts": cohorts,
                "matrix_row_keys": [copy.deepcopy(row["key"]) for row in selected],
                "native_problem_level_runnable": (
                    run_cell.get("status") == "runnable"
                    if isinstance(run_cell, dict) else False
                ),
                "native_row_passed": any(
                    cohort in {"native", "partial_native_row"}
                    for cohort in cohorts
                ),
                "reduced_proxy_only": cohorts == ["reduced_proxy"],
                "unscheduled_reason": reason,
            }
            problem_rows.append(row_record)
            row_records.append(row_record)
            schedule_counter[state] += 1
            accounted_matrix_rows += len(selected)
        problem_state_counts = dict(sorted(Counter(
            row["schedule_state"] for row in problem_rows
        ).items()))
        problem_records.append({
            "category": source["category"],
            "slug": slug,
            "architecture_reason": (
                excluded_scope[slug]["reason"] if architecture else None
            ),
            "runnability_status": (
                run_cell.get("status") if isinstance(run_cell, dict) else None
            ),
            "source_workload_row_count": len(problem_rows),
            "matrix_schedule_row_count": sum(
                len(row["matrix_row_keys"]) for row in problem_rows
            ),
            "source_rows_with_any_schedule": sum(
                bool(row["matrix_row_keys"]) for row in problem_rows
            ),
            "source_rows_unscheduled": sum(
                not row["matrix_row_keys"] for row in problem_rows
            ),
            "schedule_state_counts": problem_state_counts,
        })
    _require(accounted_matrix_rows == len(matrix["rows"]),
             "not every live matrix row was reconciled exactly once")
    return row_records, problem_records, {
        state: schedule_counter[state] for state in SCHEDULE_STATES
    }


def _direct_child_name(value: str, *, label: str) -> str:
    _require(
        isinstance(value, str) and value not in {"", ".", ".."}
        and "/" not in value and "\\" not in value
        and Path(value).name == value,
        f"{label} is not a safe direct-child name: {value!r}",
    )
    return value


def _owner_reference(name: str, payload: bytes, marker: dict[str, Any]) -> dict:
    return {
        "name": name,
        "sha256": sha256_bytes(payload),
        "owner_seal_sha256": marker["owner_seal_sha256"],
    }


def _plan_owner_document(
    *, source_commit: str, matrix_name: str, inventory_name: str,
    runnability_owner: dict[str, Any], shrink_owner: dict[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(source_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
        f"source commit is not lowercase 40-hex: {source_commit!r}",
    )
    matrix_name = _direct_child_name(matrix_name, label="matrix name")
    inventory_name = _direct_child_name(inventory_name, label="inventory name")
    _require(matrix_name != inventory_name,
             "matrix and inventory names must be distinct")
    unsigned = {
        "schema_version": 1,
        "kind": PLAN_OWNER_KIND,
        "source_commit": source_commit,
        "matrix_name": matrix_name,
        "inventory_name": inventory_name,
        "runnability_owner": copy.deepcopy(runnability_owner),
        "shrink_owner": copy.deepcopy(shrink_owner),
    }
    return {**unsigned, "owner_seal_sha256": canonical_sha256(unsigned)}


def _parse_canonical_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LiveCoverageError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LiveCoverageError(f"{label} is invalid JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    _require(payload == canonical_json_bytes(value), f"{label} is not canonical")
    return value


def initialize_plan_owner(
    *, plan_root: Path, source_commit: str, matrix_name: str,
    inventory_name: str, runnability_owner: dict[str, Any],
    shrink_owner: dict[str, Any], allow_create: bool = True,
) -> dict[str, Any]:
    """Create or verify the closed matrix+inventory publication namespace."""
    identity = runnability_tool.validate_external_campaign_root(plan_root)
    marker = _plan_owner_document(
        source_commit=source_commit, matrix_name=matrix_name,
        inventory_name=inventory_name, runnability_owner=runnability_owner,
        shrink_owner=shrink_owner,
    )
    expected = canonical_json_bytes(marker)
    names = runnability_tool.external_root_names(identity)
    if PLAN_OWNER_FILENAME not in names:
        _require(allow_create, "live plan owner marker is absent")
        _require(not names, "new live plan root must be empty")
        runnability_tool.publish_external_child(
            identity, PLAN_OWNER_FILENAME, expected
        )
        names = runnability_tool.external_root_names(identity)
    observed = runnability_tool.read_external_child(
        identity, PLAN_OWNER_FILENAME
    )
    _require(observed == expected, "live plan owner marker mismatch")
    _require(
        names <= {PLAN_OWNER_FILENAME, matrix_name, inventory_name},
        f"live plan root contains undeclared children: {sorted(names)!r}",
    )
    _require(
        inventory_name not in names or matrix_name in names,
        "live inventory exists without its matrix",
    )
    return _owner_reference(PLAN_OWNER_FILENAME, observed, marker)


def _load_owner_child(root: Path, path: Path, *, label: str) -> tuple[bytes, dict]:
    root_identity = runnability_tool.validate_external_campaign_root(root)
    candidate = Path(path)
    _require(candidate.is_absolute(), f"{label} must be an absolute path")
    _require(candidate.parent == Path(root_identity["canonical_path"]),
             f"{label} must be a direct child of its owner root")
    payload = runnability_tool.read_external_child(root_identity, candidate.name)
    return payload, root_identity


def _require_distinct_owner_roots(
    plan_identity: dict[str, Any], run_identity: dict[str, Any],
    shrink_identity: dict[str, Any],
) -> None:
    identities = (plan_identity, run_identity, shrink_identity)
    inode_keys = {
        (identity.get("device"), identity.get("inode"))
        for identity in identities
    }
    paths = [Path(str(identity.get("canonical_path", ""))) for identity in identities]
    overlaps = any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(paths)
        for right in paths[index + 1:]
    )
    _require(
        len(inode_keys) == 3 and not overlaps,
        "live plan, runnability, and shrink roots must be distinct",
    )


def build_live_inventory(
    *, repo_root: Path = REPO_ROOT, matrix_path: Path, matrix_root: Path,
    runnability_path: Path, runnability_root: Path,
    shrink_path: Path, shrink_root: Path, inventory_name: str,
) -> dict[str, Any]:
    """Re-prove all producers, then return the current completeness ledger."""
    repo_root = Path(repo_root).resolve()
    matrix_payload, matrix_owner = _load_owner_child(
        matrix_root, matrix_path, label="live supplemental matrix"
    )
    run_root_identity = runnability_tool.validate_external_campaign_root(
        Path(runnability_root)
    )
    shrink_root_identity = runnability_tool.validate_external_campaign_root(
        Path(shrink_root)
    )
    _require_distinct_owner_roots(
        matrix_owner, run_root_identity, shrink_root_identity
    )
    expected_matrix = matrix_builder.build_matrix(
        repo_root,
        runnability_path=runnability_path,
        runnability_root=runnability_root,
        shrink_path=shrink_path,
        shrink_root=shrink_root,
    )
    _require(
        matrix_payload == matrix_builder.serialize_matrix(expected_matrix),
        "live supplemental matrix bytes do not equal deterministic regeneration",
    )
    try:
        matrix_schema = json.loads((repo_root / matrix_builder.SCHEMA_PATH).read_text())
        jsonschema.validate(expected_matrix, matrix_schema)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise LiveCoverageError(f"live supplemental matrix schema failure: {exc}") from exc

    roster, _roster_sha = matrix_builder._load_roster(repo_root)
    runnability, run_sha, probe_contract = matrix_builder._load_current_runnability(
        Path(runnability_path), repo_root=repo_root, roster=roster,
        external_campaign_root=Path(runnability_root),
    )
    targets = shrink_tool.host_capacity_slugs(runnability)
    try:
        shrink_document, shrink_sha = shrink_tool.load_completed_shrink_provenance(
            Path(shrink_path), runnability=runnability,
            runnability_sha256=run_sha, probe_contract=probe_contract,
            targets=targets, runnability_path=Path(runnability_path),
            reduced_output_root=Path(shrink_root),
        )
    except (
        OSError, ValueError, runnability_tool.PrerequisiteContractError,
        jsonschema.ValidationError,
    ) as exc:
        raise LiveCoverageError(f"live shrink authority is invalid: {exc}") from exc

    benchmark_root = repo_root / historical.DEFAULT_INPUTS.benchmark_root
    problems = historical._load_benchmark_index(repo_root, benchmark_root)
    for slug, source in problems.items():
        source["rows"] = matrix_builder._workload_rows(
            repo_root / source["workload_path"]
        )
    scope_path = repo_root / historical.DEFAULT_INPUTS.port_scope
    scope, scope_payload = historical._load_port_scope(scope_path)
    included, excluded_scope = historical._validate_port_scope(scope, problems)

    row_records, problem_records, state_counts = build_row_ledger(
        problems=problems, included=included, excluded_scope=excluded_scope,
        runnability=runnability, shrink_document=shrink_document,
        matrix=expected_matrix,
    )
    architecture_rows = sum(
        len(problems[slug]["rows"]) for slug in excluded_scope
    )
    in_scope_rows = sum(len(problems[slug]["rows"]) for slug in included)
    _require(
        (len(problems), len(row_records), len(excluded_scope), architecture_rows,
         len(included), in_scope_rows)
        == (EXPECTED_PROBLEMS, EXPECTED_ROWS, EXPECTED_ARCHITECTURE_PROBLEMS,
            EXPECTED_ARCHITECTURE_ROWS, EXPECTED_IN_SCOPE_PROBLEMS,
            EXPECTED_IN_SCOPE_ROWS),
        "current SOL corpus tripwires changed",
    )
    rows_with_schedule = sum(bool(row["matrix_row_keys"]) for row in row_records)
    unscheduled_in_scope = state_counts.get("in_scope_unscheduled", 0)
    _require(
        architecture_rows + rows_with_schedule + unscheduled_in_scope
        == len(row_records),
        "row coverage identity does not close",
    )
    category_counts = {
        category: {
            "problems": sum(row["category"] == category for row in problem_records),
            "source_workload_rows": sum(
                row["category"] == category for row in row_records
            ),
            "source_rows_with_any_schedule": sum(
                row["category"] == category and bool(row["matrix_row_keys"])
                for row in row_records
            ),
            "source_rows_unscheduled": sum(
                row["category"] == category and not row["matrix_row_keys"]
                for row in row_records
            ),
        }
        for category in ("FIB", "L1", "L2", "Quant")
    }
    benchmark_manifest = [
        {"path": problems[slug]["workload_path"],
         "sha256": problems[slug]["workload_sha256"]}
        for slug in sorted(problems, key=lambda item: problems[item]["workload_path"])
    ]
    source_commit = expected_matrix["source_commit"]
    _require(
        runnability.get("source_commit") == source_commit
        and shrink_document.get("source_commit") == source_commit,
        "live evidence source commits do not equal the matrix source commit",
    )
    plan_owner = initialize_plan_owner(
        plan_root=Path(matrix_root), source_commit=source_commit,
        matrix_name=Path(matrix_path).name, inventory_name=inventory_name,
        runnability_owner=runnability["external_campaign_owner"],
        shrink_owner=shrink_document["reduced_output_owner"],
        allow_create=False,
    )
    result = {
        "schema_version": 1,
        "artifact_id": "sm86_sol_live_workload_coverage_inventory_v1",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "source_commit": source_commit,
        "claim_policy": {
            "architecture_inapplicable_is_not_native_failure": True,
            "partial_native_is_row_only": True,
            "reduced_proxy_is_not_native_predictive": True,
            "trace_completeness_requires_separate_done_evidence": True,
        },
        "inputs": {
            "matrix": {
                "name": Path(matrix_path).name,
                "sha256": sha256_bytes(matrix_payload),
                "owner": plan_owner,
            },
            "runnability": {
                "name": Path(runnability_path).name,
                "sha256": run_sha,
                "owner": runnability["external_campaign_owner"],
            },
            "shrink": {
                "name": Path(shrink_path).name,
                "sha256": shrink_sha,
                "owner": shrink_document["reduced_output_owner"],
            },
            "port_scope": {
                "path": historical.DEFAULT_INPUTS.port_scope.as_posix(),
                "sha256": sha256_bytes(scope_payload),
            },
            "benchmark_workloads": {
                "root": historical.DEFAULT_INPUTS.benchmark_root.as_posix(),
                "file_count": len(benchmark_manifest),
                "manifest_sha256": canonical_sha256(benchmark_manifest),
            },
        },
        "counts": {
            "problems": len(problems),
            "source_workload_rows": len(row_records),
            "architecture_inapplicable_problems": len(excluded_scope),
            "architecture_inapplicable_rows": architecture_rows,
            "in_scope_problems": len(included),
            "in_scope_rows": in_scope_rows,
            "matrix_schedule_rows": len(expected_matrix["rows"]),
            "source_rows_with_any_schedule": rows_with_schedule,
            "source_rows_unscheduled_in_scope": unscheduled_in_scope,
            "native_scheduled_rows": state_counts.get("native_scheduled", 0),
            "partial_native_scheduled_rows": state_counts.get(
                "partial_native_scheduled", 0
            ),
            "reduced_proxy_scheduled_rows": state_counts.get(
                "reduced_proxy_scheduled", 0
            ),
            "partial_native_and_reduced_proxy_scheduled_rows": state_counts.get(
                "partial_native_and_reduced_proxy_scheduled", 0
            ),
        },
        "schedule_state_counts": state_counts,
        "by_category": category_counts,
        "problems": problem_records,
        "rows": row_records,
    }
    try:
        schema = json.loads((repo_root / SCHEMA_PATH).read_text())
        jsonschema.validate(result, schema)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise LiveCoverageError(f"live coverage output violates schema: {exc}") from exc
    return result


def initialize_plan_root_from_authorities(
    *, repo_root: Path = REPO_ROOT, plan_root: Path, matrix_name: str,
    inventory_name: str, runnability_path: Path, runnability_root: Path,
    shrink_path: Path, shrink_root: Path,
) -> dict[str, Any]:
    """Validate completed authorities before creating the closed plan marker."""
    root = Path(repo_root).resolve()
    plan_identity = runnability_tool.validate_external_campaign_root(plan_root)
    run_identity = runnability_tool.validate_external_campaign_root(
        runnability_root
    )
    shrink_identity = runnability_tool.validate_external_campaign_root(
        shrink_root
    )
    _require_distinct_owner_roots(plan_identity, run_identity, shrink_identity)
    roster, _roster_sha = matrix_builder._load_roster(root)
    runnability, run_sha, probe_contract = matrix_builder._load_current_runnability(
        Path(runnability_path), repo_root=root, roster=roster,
        external_campaign_root=Path(runnability_root),
    )
    try:
        shrink_document, _shrink_sha = shrink_tool.load_completed_shrink_provenance(
            Path(shrink_path), runnability=runnability,
            runnability_sha256=run_sha, probe_contract=probe_contract,
            targets=shrink_tool.host_capacity_slugs(runnability),
            runnability_path=Path(runnability_path),
            reduced_output_root=Path(shrink_root),
        )
    except (
        OSError, ValueError, runnability_tool.PrerequisiteContractError,
        jsonschema.ValidationError,
    ) as exc:
        raise LiveCoverageError(f"live shrink authority is invalid: {exc}") from exc
    source_commit = runnability.get("source_commit")
    _require(
        isinstance(source_commit, str)
        and shrink_document.get("source_commit") == source_commit,
        "live authority source commits differ",
    )
    return initialize_plan_owner(
        plan_root=plan_root, source_commit=source_commit,
        matrix_name=matrix_name, inventory_name=inventory_name,
        runnability_owner=runnability["external_campaign_owner"],
        shrink_owner=shrink_document["reduced_output_owner"],
        allow_create=True,
    )


def _deliver(
    document: dict[str, Any], *, root: Path, emit: Path | None,
    check: Path | None,
) -> int:
    _require((emit is None) != (check is None),
             "exactly one of --emit/--check is required")
    identity = runnability_tool.validate_external_campaign_root(root)
    path = emit if emit is not None else check
    assert path is not None
    _require(path.is_absolute() and path.parent == Path(identity["canonical_path"]),
             "live coverage output must be a direct child of --inventory-root")
    matrix_input = document["inputs"]["matrix"]
    marker_ref = matrix_input["owner"]
    _require(
        marker_ref.get("name") == PLAN_OWNER_FILENAME,
        "live plan owner reference name is invalid",
    )
    marker_payload = runnability_tool.read_external_child(
        identity, PLAN_OWNER_FILENAME
    )
    marker = _parse_canonical_object(marker_payload, label="live plan owner")
    _require(
        marker_ref == _owner_reference(PLAN_OWNER_FILENAME, marker_payload, marker)
        and marker.get("kind") == PLAN_OWNER_KIND
        and marker.get("matrix_name") == matrix_input["name"]
        and marker.get("inventory_name") == path.name
        and marker.get("source_commit") == document["source_commit"],
        "live plan owner reference does not bind the output pair",
    )
    matrix_payload = runnability_tool.read_external_child(
        identity, matrix_input["name"]
    )
    _require(
        sha256_bytes(matrix_payload) == matrix_input["sha256"],
        "live plan matrix bytes do not match the inventory input binding",
    )
    expected_before = {PLAN_OWNER_FILENAME, matrix_input["name"]}
    expected_after = expected_before | {path.name}
    names = runnability_tool.external_root_names(identity)
    _require(
        names == (expected_after if check is not None else expected_before),
        f"live plan namespace closure mismatch: {sorted(names)!r}",
    )
    expected = canonical_json_bytes(document)
    if check is not None:
        observed = runnability_tool.read_external_child(identity, check.name)
        marker_after = runnability_tool.read_external_child(
            identity, PLAN_OWNER_FILENAME
        )
        matrix_after = runnability_tool.read_external_child(
            identity, matrix_input["name"]
        )
        observed_after = runnability_tool.read_external_child(
            identity, check.name
        )
        names_after = runnability_tool.external_root_names(identity)
        _require(
            marker_after == marker_payload
            and matrix_after == matrix_payload
            and sha256_bytes(matrix_after) == matrix_input["sha256"]
            and observed_after == observed
            and names_after == names,
            "live plan owner, matrix, inventory, or namespace changed "
            "during inventory check",
        )
        if observed != expected:
            print(f"live SOL coverage inventory byte mismatch: {check}", file=sys.stderr)
            return 1
        print(
            "live SOL coverage inventory verified: "
            f"rows={document['counts']['source_workload_rows']} "
            f"scheduled={document['counts']['source_rows_with_any_schedule']}"
        )
        return 0
    runnability_tool.publish_external_child(identity, emit.name, expected)
    marker_after = runnability_tool.read_external_child(
        identity, PLAN_OWNER_FILENAME
    )
    matrix_after = runnability_tool.read_external_child(
        identity, matrix_input["name"]
    )
    observed = runnability_tool.read_external_child(identity, emit.name)
    marker_final = runnability_tool.read_external_child(
        identity, PLAN_OWNER_FILENAME
    )
    matrix_final = runnability_tool.read_external_child(
        identity, matrix_input["name"]
    )
    observed_final = runnability_tool.read_external_child(identity, emit.name)
    names_after = runnability_tool.external_root_names(identity)
    _require(
        marker_after == marker_payload
        and marker_final == marker_payload
        and matrix_after == matrix_payload
        and matrix_final == matrix_payload
        and sha256_bytes(matrix_after) == matrix_input["sha256"]
        and sha256_bytes(matrix_final) == matrix_input["sha256"]
        and observed == expected
        and observed_final == expected
        and names_after == expected_after,
        "live plan owner, matrix, inventory, or namespace changed "
        "during inventory publication",
    )
    print(
        "live SOL coverage inventory: "
        f"rows={document['counts']['source_workload_rows']} "
        f"scheduled={document['counts']['source_rows_with_any_schedule']} -> {emit}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize-plan-root", action="store_true")
    mode.add_argument("--emit", type=Path)
    mode.add_argument("--check", type=Path)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--inventory-name")
    parser.add_argument("--runnability", type=Path, required=True)
    parser.add_argument("--runnability-root", type=Path, required=True)
    parser.add_argument("--shrink-provenance", type=Path, required=True)
    parser.add_argument("--shrink-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.initialize_plan_root:
            _require(args.inventory_name is not None,
                     "--inventory-name is required for plan initialization")
            identity = runnability_tool.validate_external_campaign_root(
                args.inventory_root
            )
            _require(
                args.inventory_root == args.matrix_root
                and args.matrix.is_absolute()
                and args.matrix.parent == Path(identity["canonical_path"]),
                "initialized plan root must own the declared matrix path",
            )
            owner = initialize_plan_root_from_authorities(
                plan_root=args.inventory_root,
                matrix_name=args.matrix.name,
                inventory_name=args.inventory_name,
                runnability_path=args.runnability,
                runnability_root=args.runnability_root,
                shrink_path=args.shrink_provenance,
                shrink_root=args.shrink_root,
            )
            print(
                "live SOL coverage plan root initialized: "
                f"{args.inventory_root} owner={owner['sha256']}"
            )
            return 0
        output = args.emit if args.emit is not None else args.check
        assert output is not None
        _require(args.inventory_name is None,
                 "--inventory-name is only valid with --initialize-plan-root")
        _require(args.inventory_root == args.matrix_root,
                 "matrix and live inventory must share the dedicated plan root")
        document = build_live_inventory(
            matrix_path=args.matrix, matrix_root=args.matrix_root,
            runnability_path=args.runnability,
            runnability_root=args.runnability_root,
            shrink_path=args.shrink_provenance, shrink_root=args.shrink_root,
            inventory_name=output.name,
        )
        return _deliver(
            document, root=args.inventory_root,
            emit=args.emit, check=args.check,
        )
    except (
        OSError, LiveCoverageError, matrix_builder.WorkloadMatrixError,
        runnability_tool.PrerequisiteContractError,
    ) as exc:
        print(f"sol_live_workload_coverage_inventory: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
