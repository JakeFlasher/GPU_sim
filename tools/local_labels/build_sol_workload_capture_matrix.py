#!/usr/bin/env python3
"""Build the additive, workload-row-granular SOL capture matrix.

This artifact expands only runnable SOL entries from the canonical sm_86
capture matrix.  It never edits or replaces the canonical 344-cell authority.
Native workload rows and reduced proxy rows retain different claim scopes and
are never placed in the same chunk.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import sol_reconstitution as sr  # noqa: E402
from tools.local_labels import sol_shrink_apply as shrink  # noqa: E402

CANONICAL_MATRIX_PATH = Path(
    "data/local_labels/campaign/sm86_reference_capture_matrix.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/local_labels/campaign/sol_workload_capture_matrix.json"
)
SCHEMA_PATH = Path(
    "workloads/simbench_bw/schemas/sol_workload_capture_matrix.schema.json"
)
SHRINK_PROVENANCE_PATH = Path(
    "data/local_labels/campaign/sol_shrink_provenance_sm86.json"
)
RUNNABILITY_PATH = Path("data/local_labels/campaign/sol_runnability_sm86.json")
ROSTER_PATH = Path(
    "tools/sol_baselines_harness/calibration/sm86_coverage_ledger_v1.toml"
)
PROBE_CONFIG_PATH = Path("tools/local_labels/sol_full_correctness_config.v1.json")
TRACE_BUDGETS_PATH = Path("workloads/simbench_bw/trace_budgets.toml")
SUPPLEMENTAL_BUDGET_POLICY_PATH = Path(
    "tools/local_labels/sol_supplemental_trace_budget_policy.v1.json"
)
LIVE_NATIVE_ENV_FINGERPRINT_PATH = Path(
    "tools/local_labels/authorities"
    "/sol_native_env_fingerprint_sm86.live.v2.json"
)
LIVE_NATIVE_ENV_FINGERPRINT_SCHEMA_PATH = Path(
    "workloads/simbench_bw/schemas"
    "/sol_native_env_fingerprint.live.v2.schema.json"
)
SOL_EXECUTABLE_PATH = Path("tools/sol_execbench/.venv/bin/sol-execbench")
SOL_BENCHMARK_ROOT = Path("tools/sol_execbench/data/benchmark")
SOL_BASELINES_ROOT = Path("workloads/sol_execbench_baselines")
SUBSET_BY_CATEGORY = {
    "L1": "L1", "L2": "L2", "FIB": "FlashInfer-Bench", "Quant": "Quant",
}
ROWS_PER_CHUNK = 16
LIVE_PLAN_OWNER_FILENAME = ".sol-live-coverage-plan-owner-v1.json"
LIVE_PLAN_OWNER_KIND = "sol_live_coverage_plan_owner_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}|[0-9a-f]{32})$"
)
COHORT_ORDER = {"native": 0, "reduced_proxy": 1}
COHORT_ORDER["partial_native_row"] = 1
COHORT_ORDER["reduced_proxy"] = 2
REDUCED_PROXY_CLAIM_SCOPE = "reduced_proxy_bundle_only"
REDUCED_PROXY_COHORT_POLICY = (
    "reduced_proxy_bundles_only_not_native_predictive"
)
EXACT_SOL_ENVIRONMENT = {
    "CUTLASS_DIR": "third_party/cutlass",
    "PATH_prepend": "tools/sol_execbench/.venv/bin",
    "CUDA_HOME": "/usr/local/cuda-13.2",
    "NVCC_CCBIN": "g++-14",
    "CC": "gcc-14",
    "CXX": "g++-14",
    "CUDAHOSTCXX": "g++-14",
}
LAUNCH_COMPLETENESS_CONTRACT = {
    "inventory_policy": "requires_inventory_before_capture",
    "launch_scope": "all_in_profiler_roi",
    "trace_requirement": "every_inventory_launch_complete",
    "ncu_label_requirement": "every_inventory_launch_complete",
    "control_requirement": "every_inventory_launch_complete",
    "admission_policy": "all_three_layers_complete_for_exact_inventory",
}
SUPPLEMENTAL_TRACE_CARDINALITY_CONTRACT = (
    "one_selected_workload_candidate_call_all_dynamic_launches_inventoried"
)
SUPPLEMENTAL_CAPTURE_TARGET = {
    "source_workload_index": 0,
    "staged_workload_index": 0,
    "selected_candidate_call_ordinal": 0,
    "candidate_user_fn_calls": 1,
    "dynamic_launch_scope": "all_in_profiler_roi",
    "dynamic_launch_count_source": "runtime_inventory",
    "profiler_range": "cudaProfilerStartStop",
    "setup_outside_profiler_range": True,
    "timing_methodology": "cuda_events",
}
SUPPLEMENTAL_CAPTURE_LAUNCH_CONTRACT = {
    "schema_version": 2,
    "label_driver_warmup_count": 1,
    "label_effective_warmup_count": 0,
    "label_launch_index": 0,
    "label_measured_reps": 1,
    "trace_candidate_call_count": 1,
    "trace_dynamic_launch_scope": "all_in_profiler_roi",
    "trace_dynamic_launch_count_source": "runtime_inventory",
}


class WorkloadMatrixError(RuntimeError):
    """A fail-closed source, identity, or serialization error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def tree_sha256(path: Path) -> str:
    """Match the canonical capture builder's sorted tree-identity algorithm."""
    if not path.is_dir():
        raise WorkloadMatrixError(f"required directory absent: {path}")
    rows: list[dict[str, str]] = []
    for child in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            rows.append({"path": relative, "symlink": os.readlink(child)})
        elif child.is_file():
            rows.append({"path": relative, "sha256": sha256_file(child)})
    if not rows:
        raise WorkloadMatrixError(f"required directory is empty: {path}")
    return canonical_sha256(rows)


def canonical_source_sha256(value: Any) -> str:
    """Reproduce the canonical matrix builder's existing seal algorithm."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def producer_canonical_bytes(value: Any) -> bytes:
    """Match the geometry-proxy producer's identity serialization."""
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def producer_variant_id(
    *, slug: str, source_uuid: str, native_sha256: str,
    reduced_sha256: str, native_definition_sha256: str,
    reduced_definition_sha256: str | None, model_id: str,
) -> str:
    return sha256_bytes(producer_canonical_bytes({
        "model_id": model_id,
        "native_definition_sha256": native_definition_sha256,
        "native_row_sha256": native_sha256,
        "outcome": "proxy_planned",
        "reduced_definition_sha256": reduced_definition_sha256,
        "reduced_row_sha256": reduced_sha256,
        "slug": slug,
        "source_workload_uuid": source_uuid,
        "variant_contract_version": 2,
    }))


def serialize_matrix(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkloadMatrixError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadMatrixError(f"{label} must be a JSON object: {path}")
    return value


def _repo_file(
    repo_root: Path, raw: Any, *, label: str, allow_external: bool = False,
) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw:
        raise WorkloadMatrixError(f"{label} path is absent")
    declared = Path(raw)
    if declared.is_absolute():
        if not allow_external:
            raise WorkloadMatrixError(f"{label} must be repository-relative: {raw!r}")
        if declared.is_symlink():
            raise WorkloadMatrixError(f"{label} is a symbolic link: {raw}")
        path = declared.resolve()
        if not path.is_file():
            raise WorkloadMatrixError(f"{label} is not a regular file: {raw}")
        return path, str(path)
    if declared == Path(".") or ".." in declared.parts:
        raise WorkloadMatrixError(f"{label} must be repository-relative: {raw!r}")
    root = repo_root.resolve()
    lexical_path = root / declared
    if lexical_path.is_symlink():
        raise WorkloadMatrixError(f"{label} is a symbolic link: {raw}")
    path = lexical_path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkloadMatrixError(f"{label} escapes repository: {raw!r}") from exc
    if not path.is_file():
        raise WorkloadMatrixError(f"{label} is not a regular file: {raw}")
    return path, declared.as_posix()


def _explicit_input_file(
    repo_root: Path, value: Path | None, default: Path, *, label: str,
) -> Path:
    declared = default if value is None else Path(value)
    raw = str(declared if declared.is_absolute() else declared.as_posix())
    path, _reference = _repo_file(
        repo_root, raw, label=label, allow_external=True,
    )
    return path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _live_native_env_pin(repo_root: Path) -> dict[str, str]:
    """Validate and bind the live authority without touching frozen v1 bytes."""
    path, reference = _repo_file(
        repo_root, LIVE_NATIVE_ENV_FINGERPRINT_PATH.as_posix(),
        label="live SOL native-environment fingerprint",
    )
    schema_path, _schema_reference = _repo_file(
        repo_root, LIVE_NATIVE_ENV_FINGERPRINT_SCHEMA_PATH.as_posix(),
        label="live SOL native-environment fingerprint schema",
    )
    document = _load_object(
        path, label="live SOL native-environment fingerprint"
    )
    try:
        schema = json.loads(schema_path.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(document, schema)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError,
            jsonschema.ValidationError) as exc:
        raise WorkloadMatrixError(
            f"live SOL native-environment fingerprint is invalid: {exc}"
        ) from exc
    components = document["components"]
    derived = canonical_sha256(components)
    if (
        document["native_env_hash"] != derived
        or sr.certified_native_env_hash(path) != derived
    ):
        raise WorkloadMatrixError(
            "live SOL native-environment fingerprint does not rederive"
        )
    return {
        "path": reference,
        "sha256": sha256_file(path),
        "native_env_hash": derived,
    }


def _validate_variant_native_env_closure(
    variants: list[dict[str, Any]], *, expected_hash: str,
) -> None:
    if not variants:
        raise WorkloadMatrixError("supplemental matrix has no variants")
    for variant in variants:
        adapter = variant.get("adapter_contract")
        invocation_adapter = (variant.get("invocation") or {}).get(
            "adapter_contract"
        )
        if (
            variant.get("native_env_hash") != expected_hash
            or not isinstance(adapter, dict)
            or adapter.get("native_env_hash") != expected_hash
            or not isinstance(invocation_adapter, dict)
            or invocation_adapter.get("native_env_hash") != expected_hash
            or adapter != invocation_adapter
            or variant.get("adapter_contract_sha256")
            != canonical_sha256(adapter)
        ):
            raise WorkloadMatrixError(
                "supplemental variant does not close over the live fingerprint: "
                f"{variant.get('slug')}:{variant.get('variant_id')}"
            )


def _load_roster(repo_root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    path = repo_root / ROSTER_PATH
    try:
        document = tomllib.loads(path.read_text())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise WorkloadMatrixError(f"cannot load SOL roster: {exc}") from exc
    rows = document.get("rows")
    meta = document.get("meta")
    if (
        document.get("schema_version") != 1
        or not isinstance(meta, dict)
        or not isinstance(rows, list)
        or meta.get("row_count") != len(rows)
    ):
        raise WorkloadMatrixError("SOL roster accounting is invalid")
    by_slug: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise WorkloadMatrixError("SOL roster row is malformed")
        slug = row.get("slug")
        if (
            not isinstance(slug, str) or not slug
            or slug in by_slug
            or row.get("category") not in SUBSET_BY_CATEGORY
            or not isinstance(row.get("language"), str)
            or not row["language"]
        ):
            raise WorkloadMatrixError("SOL roster identity is invalid")
        by_slug[slug] = row
    return by_slug, sha256_file(path)


def _validate_completed_runnability(
    document: dict[str, Any], *, repo_root: Path,
    roster: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if repo_root.resolve() != REPO_ROOT.resolve():
        raise WorkloadMatrixError(
            "completed SOL runnability validation requires the canonical repository"
        )
    try:
        fingerprint = sr.native_env_fingerprint()
        contract = sr.build_probe_contract(
            native_env_hash=fingerprint["hash"],
            outer_timeout=document.get("outer_timeout_seconds"),
            target_cells=list(roster.values()),
        )
        sr.validate_completed_runnability(
            document,
            expected_contract=contract,
            expected_cells=list(roster.values()),
        )
        return contract
    except (KeyError, TypeError, sr.PrerequisiteContractError) as exc:
        raise WorkloadMatrixError(
            f"SOL runnability authority is not the complete current contract: {exc}"
        ) from exc


def _load_current_runnability(
    path: Path, *, repo_root: Path, roster: dict[str, dict[str, Any]],
    external_campaign_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Load exact authority bytes through the shared current-contract gate."""
    try:
        artifact, payload_sha = sr.load_completed_runnability(
            path, expected_contract=None, require_external_owner=True,
            external_campaign_root=external_campaign_root,
        )
    except (OSError, ValueError, sr.PrerequisiteContractError) as exc:
        raise WorkloadMatrixError(
            f"SOL runnability authority exact-byte load failed: {exc}"
        ) from exc
    contract = _validate_completed_runnability(
        artifact, repo_root=repo_root, roster=roster,
    )
    try:
        sr.validate_completed_runnability(
            artifact, expected_contract=contract,
            expected_cells=list(roster.values()), require_external_owner=True,
            external_campaign_root=external_campaign_root,
        )
    except sr.PrerequisiteContractError as exc:
        raise WorkloadMatrixError(
            f"SOL runnability authority current-contract mismatch: {exc}"
        ) from exc
    return artifact, payload_sha, contract


def _flashinfer_contracts(
    probe_contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        return {
            category: sr.flashinfer_capture_contract(
                probe_contract, category=category,
            )
            for category in SUBSET_BY_CATEGORY
        }
    except sr.PrerequisiteContractError as exc:
        raise WorkloadMatrixError(
            f"verified FlashInfer capture contract is unavailable: {exc}"
        ) from exc


def _load_budget_policy(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    policy_path = repo_root / SUPPLEMENTAL_BUDGET_POLICY_PATH
    policy = _load_object(policy_path, label="SOL supplemental budget policy")
    expected_policy = {
        "schema_version": 1,
        "policy_id": "sm86_sol_supplemental_unsampled_v1",
        "default_max_trace_bytes": 17179869184,
        "default_max_wall_time_s": 7200,
        "sampling_policy": "off",
        "derivation": "explicit_slug_or_policy_default_ceiling",
        "breach_disposition": "hard_refusal_preserve_complete_trace",
        "partial_trace_allowed": False,
        "source_basis": "supplemental_raw_staging_ceiling_full_trace_retention",
    }
    if policy != expected_policy:
        raise WorkloadMatrixError("SOL supplemental budget policy drift")
    source_path = repo_root / TRACE_BUDGETS_PATH
    try:
        source = tomllib.loads(source_path.read_text())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise WorkloadMatrixError(f"cannot load trace budgets: {exc}") from exc
    rows: dict[str, dict[str, Any]] = {}
    for row in source.get("cells", []):
        if not isinstance(row, dict) or not isinstance(row.get("cell_id"), str):
            raise WorkloadMatrixError("trace-budget row is malformed")
        slug = row["cell_id"]
        if slug in rows:
            raise WorkloadMatrixError(f"duplicate trace-budget slug: {slug}")
        rows[slug] = row
    policy["path"] = SUPPLEMENTAL_BUDGET_POLICY_PATH.as_posix()
    policy["sha256"] = sha256_file(policy_path)
    policy["trace_budgets_path"] = TRACE_BUDGETS_PATH.as_posix()
    policy["trace_budgets_sha256"] = sha256_file(source_path)
    return policy, rows


def _trace_budget(
    slug: str, *, policy: dict[str, Any], explicit: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = explicit.get(slug)
    source_bytes = source.get("max_trace_bytes") if source else None
    source_time = source.get("max_wall_time_s") if source else None
    if source is not None and (
        type(source_bytes) is not int or source_bytes <= 0
        or type(source_time) is not int or source_time <= 0
        or not isinstance(source.get("sampling_policy"), str)
    ):
        raise WorkloadMatrixError(f"invalid explicit trace budget: {slug}")
    return {
        "status": "declared",
        "policy_path": policy["path"],
        "policy_sha256": policy["sha256"],
        "trace_budgets_path": policy["trace_budgets_path"],
        "trace_budgets_sha256": policy["trace_budgets_sha256"],
        "explicit_slug_budget_sha256": (
            canonical_sha256(source) if source is not None else None
        ),
        "derivation": policy["derivation"],
        "max_trace_bytes": (
            source_bytes
            if source_bytes is not None
            else policy["default_max_trace_bytes"]
        ),
        "max_wall_time_s": (
            source_time
            if source_time is not None
            else policy["default_max_wall_time_s"]
        ),
        "sampling_policy": "off",
        "breach_disposition": policy["breach_disposition"],
        "partial_trace_allowed": False,
    }


def partial_native_variant_id(
    *, slug: str, uuid: str, row_sha256: str, diagnostic_sha256: str,
    cell_sha256: str, resume_identity_sha256: str,
) -> str:
    return canonical_sha256({
        "cell_sha256": cell_sha256,
        "diagnostic_sha256": diagnostic_sha256,
        "identity_version": 1,
        "resume_identity_sha256": resume_identity_sha256,
        "row_sha256": row_sha256,
        "slug": slug,
        "source_workload_uuid": uuid,
        "variant_kind": "partial_native_row",
    })


def _control_contract() -> dict[str, Any]:
    nvcc = Path("/usr/local/cuda-13.2/bin/nvcc")
    cuobjdump = Path("/usr/local/cuda-13.2/bin/cuobjdump")
    for path in (nvcc, cuobjdump):
        if path.is_symlink() or not path.is_file():
            raise WorkloadMatrixError(f"SOL static-control tool is absent: {path}")
    return {
        "policy": "post_trace_module_or_typed",
        "module_discovery": "nvbit_runtime_module_inventory",
        "required_outcomes": [
            "exact_accepted_launch_runtime_binary_controls_emitted",
            "exact_accepted_launch_static_rebind_controls_emitted",
            "typed_unavailable_no_stable_static_module",
        ],
        "static_rebind": {
            "policy": "preserved_module_exact_function",
            "toolkit_version": "V13.2",
            "allowed_code_arches": ["sm_80", "sm_86"],
            "nvcc_path": str(nvcc),
            "nvcc_sha256": sha256_file(nvcc),
            "cuobjdump_path": str(cuobjdump),
            "cuobjdump_sha256": sha256_file(cuobjdump),
            "ncu_report_binding": "raw_text_section_exactly_once",
        },
    }


def _trace_status(trace: dict[str, Any]) -> Any:
    status = trace.get("status")
    if status is None and isinstance(trace.get("evaluation"), dict):
        status = trace["evaluation"].get("status")
    if isinstance(status, dict):
        status = status.get("value")
    return status


def _executor_effective_workload(source: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the pinned executor's JSON view of one source workload.

    ``sol-execbench`` parses each JSONL row through ``Workload`` and writes
    ``model_dump(mode="json")`` into its trace.  That materializes the five
    ``ToleranceSpec`` defaults, so the trace object is deliberately not the
    sparse source object.  The L2 corpus also carries the historical misspelling
    ``required_match_ratio``; the pinned model ignores that key and executes
    with the stricter ``required_matched_ratio=0.99`` default.  Preserve that
    observed execution truth without weakening the exact comparison for any
    other workload field.
    """
    effective = copy.deepcopy(source)
    tolerance = effective.setdefault("tolerance", {})
    if not isinstance(tolerance, dict):
        raise WorkloadMatrixError("SOL source workload tolerance is not an object")
    if (
        "required_match_ratio" in tolerance
        and "required_matched_ratio" in tolerance
    ):
        raise WorkloadMatrixError(
            "SOL source workload has ambiguous matched-ratio spellings"
        )
    tolerance.pop("required_match_ratio", None)
    for field, default in (
        ("max_atol", 1e-2),
        ("max_rtol", 1e-2),
        ("required_matched_ratio", 0.99),
        ("max_error_cap", None),
        ("allow_negative_inf", False),
    ):
        tolerance.setdefault(field, default)
    return effective


def _native_problem_contract(
    *, repo_root: Path, record: dict[str, Any], roster_row: dict[str, Any],
    runnability: dict[str, Any], runnability_path: Path,
    policy: dict[str, Any], explicit_budgets: dict[str, dict[str, Any]],
    control_bits: dict[str, Any],
    flashinfer_contracts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slug = roster_row["slug"]
    category = roster_row["category"]
    language = roster_row["language"]
    solution_rel = (
        SOL_BASELINES_ROOT / category / slug / language / "solution.json"
    )
    solution_path, solution_ref = _repo_file(
        repo_root, solution_rel.as_posix(), label=f"{slug} solution"
    )
    solution = _load_object(solution_path, label=f"{slug} solution")
    definition_name = solution.get("definition")
    solution_name = solution.get("name")
    if (
        not isinstance(definition_name, str) or not definition_name
        or not isinstance(solution_name, str) or not solution_name
        or "/" in definition_name or ".." in definition_name
    ):
        raise WorkloadMatrixError(f"SOL solution identity is invalid: {slug}")
    benchmark_rel = (
        SOL_BENCHMARK_ROOT / SUBSET_BY_CATEGORY[category] / definition_name
    )
    definition_path, definition_ref = _repo_file(
        repo_root, (benchmark_rel / "definition.json").as_posix(),
        label=f"{slug} definition",
    )
    workload_path, workload_ref = _repo_file(
        repo_root, (benchmark_rel / "workload.jsonl").as_posix(),
        label=f"{slug} workload",
    )
    definition = _load_object(definition_path, label=f"{slug} definition")
    if definition.get("name") != definition_name:
        raise WorkloadMatrixError(f"SOL definition identity mismatch: {slug}")
    source_rows = _workload_rows(workload_path)
    diagnostics = record.get("workload_diagnostics")
    unexpected = record.get("unexpected_workload_diagnostics")
    if (
        not isinstance(diagnostics, list)
        or len(diagnostics) != len(source_rows)
        or not isinstance(unexpected, list)
        or unexpected
        or record.get("expected_workloads") != len(source_rows)
        or sha256_bytes(record.get("stdout", "").encode("utf-8"))
        != record.get("stdout_sha256")
        or sha256_bytes(record.get("stderr", "").encode("utf-8"))
        != record.get("stderr_sha256")
    ):
        raise WorkloadMatrixError(f"SOL row diagnostics are incomplete: {slug}")
    passed_rows: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for index, (source_row, diagnostic) in enumerate(zip(source_rows, diagnostics)):
        if not isinstance(diagnostic, dict):
            raise WorkloadMatrixError(f"SOL diagnostic is malformed: {slug}:{index}")
        trace = diagnostic.get("trace")
        trace_workload = trace.get("workload") if isinstance(trace, dict) else None
        diagnostic_sha = canonical_sha256(diagnostic)
        identity_matches = (
            diagnostic.get("workload_index") == index
            and diagnostic.get("source_line") == source_row["source_line"]
            and diagnostic.get("uuid") == source_row["source_workload_uuid"]
            and diagnostic.get("axes") == source_row["source_axes"]
            and diagnostic.get("workload_sha256") == source_row["source_row_sha256"]
        )
        if not identity_matches:
            raise WorkloadMatrixError(f"SOL diagnostic/source mismatch: {slug}:{index}")
        if diagnostic.get("diagnostic_present") is True:
            if (
                type(diagnostic.get("trace_index")) is not int
                or not isinstance(trace, dict)
                or trace_workload
                != _executor_effective_workload(source_row["source_value"])
                or trace.get("definition") != definition_name
                or trace.get("solution") != solution_name
                or diagnostic.get("status") != _trace_status(trace)
            ):
                raise WorkloadMatrixError(
                    f"SOL diagnostic trace binding mismatch: {slug}:{index}"
                )
            if str(diagnostic.get("status")).lower() == "passed":
                passed_rows.append((source_row, diagnostic, diagnostic_sha))
        elif any(
            diagnostic.get(field) is not None
            for field in ("trace_index", "status", "trace")
        ):
            raise WorkloadMatrixError(f"SOL absent diagnostic is populated: {slug}:{index}")
    if record.get("passed_workloads") != len(passed_rows):
        raise WorkloadMatrixError(f"SOL passed-workload arithmetic mismatch: {slug}")
    computed_complete = (
        record.get("returned_workloads") == len(source_rows)
        and all(row.get("diagnostic_present") is True for row in diagnostics)
        and {
            row.get("trace_index") for row in diagnostics
        } == set(range(len(source_rows)))
    )
    if record.get("diagnostics_complete") is not computed_complete:
        raise WorkloadMatrixError(f"SOL diagnostic completeness mismatch: {slug}")
    problem_level = record.get("status") == "runnable"
    if problem_level and (
        not computed_complete
        or len(passed_rows) != len(source_rows)
        or record.get("returncode") != 0
        or record.get("detail") != f"{len(source_rows)}/{len(source_rows)} workloads passed"
    ):
        raise WorkloadMatrixError(f"SOL runnable record is not complete: {slug}")
    if not problem_level and computed_complete and len(passed_rows) == len(source_rows):
        raise WorkloadMatrixError(
            f"SOL non-runnable record has an all-PASSED UUID bijection: {slug}"
        )
    if not passed_rows:
        return [], []
    expected_command = [
        "tools/sol_execbench/.venv/bin/python",
        "-m", "sol_execbench.cli.main",
        "--definition", definition_ref,
        "--workload", workload_ref,
        "--solution", solution_ref,
        "--json",
        "--config", PROBE_CONFIG_PATH.as_posix(),
        "--timeout", "1500",
    ]
    if record.get("command") != expected_command:
        raise WorkloadMatrixError(f"SOL probe command binding mismatch: {slug}")

    executable_path, executable_ref = _repo_file(
        repo_root, SOL_EXECUTABLE_PATH.as_posix(), label="SOL adapter executable"
    )
    forwarded = [benchmark_rel.as_posix(), "--solution", solution_ref, "--json"]
    flashinfer_corpus = flashinfer_contracts.get(category)
    if not isinstance(flashinfer_corpus, dict):
        raise WorkloadMatrixError(
            f"SOL FlashInfer corpus contract is absent: {slug}:{category}"
        )
    adapter_contract = {
        "contract_version": 3,
        "forwarded_argv": forwarded,
        "environment": copy.deepcopy(EXACT_SOL_ENVIRONMENT),
        "native_env_hash": runnability["native_env_hash"],
        "flashinfer_corpus": copy.deepcopy(flashinfer_corpus),
        "shape_forwarded_to_target": False,
        "label_driver_bridge_status": "implemented",
        "persistent_stage_status": "implemented",
        "capture_request_environment": "SOL_EXECBENCH_CAPTURE_REQUEST",
        "label_driver_bridge_rule": (
            "forward adapter_contract.forwarded_argv exactly without appending "
            "a synthetic --shape-point token"
        ),
    }
    capture_target = copy.deepcopy(SUPPLEMENTAL_CAPTURE_TARGET)
    cell_sha = canonical_sha256(record)
    evidence = {
        "source_kind": (
            "current_full_correctness" if problem_level
            else "current_partial_native_row"
        ),
        "source_path": runnability_path.name,
        "source_sha256": sha256_file(runnability_path),
        "cell_sha256": cell_sha,
        "resume_identity_sha256": runnability["resume_identity_sha256"],
        "probe_protocol_sha256": runnability["probe_protocol_sha256"],
        "probe_config_sha256": runnability["probe_config_sha256"],
        "sol_execbench_package_sha256": runnability[
            "sol_execbench_package_sha256"
        ],
        "native_env_hash": runnability["native_env_hash"],
        "problem_status": record["status"],
        "problem_returncode": record["returncode"],
        "diagnostics_complete": record["diagnostics_complete"],
        "expected_workloads": len(source_rows),
        "passed_workloads": len(passed_rows),
    }
    invocation = {
        "adapter_kind": "sol_execbench",
        "authority_reference_path": solution_ref,
        "authority_reference_sha256": sha256_file(solution_path),
        "executable_path": executable_ref,
        "executable_sha256": sha256_file(executable_path),
        "label_app_args": forwarded,
        "trace_app_args": forwarded,
        "evaluation_cardinality_contract": "all_effective_workloads_passed_exact",
        "trace_cardinality_contract": SUPPLEMENTAL_TRACE_CARDINALITY_CONTRACT,
        "capture_launch_contract": copy.deepcopy(
            SUPPLEMENTAL_CAPTURE_LAUNCH_CONTRACT
        ),
        "capture_target": capture_target,
        "logical_workload": {
            "kind": "sol_execbench_workload_set",
            "value": "workloads",
            "workload_scope": "native",
            "benchmark_definition": definition_name,
            "definition_file": definition_ref,
            "definition_file_sha256": sha256_file(definition_path),
            "benchmark_dir": benchmark_rel.as_posix(),
            "benchmark_tree_sha256": tree_sha256(repo_root / benchmark_rel),
            "workload_file": workload_ref,
            "workload_file_sha256": sha256_file(workload_path),
            "expected_workloads": len(source_rows),
            "evaluation_evidence": evidence,
        },
        "shape_delivery": "adapter_consumes_logical_shape",
        "target_processes": "all",
        "adapter_contract": adapter_contract,
    }
    budget = _trace_budget(slug, policy=policy, explicit=explicit_budgets)
    base = {
        "slug": slug,
        "category": category,
        "launch_scope": "all_in_profiler_roi",
        "launch_completeness_contract": copy.deepcopy(
            LAUNCH_COMPLETENESS_CONTRACT
        ),
        "workload_scope": "native",
        "source_workload_file": workload_ref,
        "source_workload_file_sha256": sha256_file(workload_path),
        "source_workload_count": len(source_rows),
        "definition_file": definition_ref,
        "definition_file_sha256": sha256_file(definition_path),
        "solution_file": solution_ref,
        "solution_file_sha256": sha256_file(solution_path),
        "adapter_contract": adapter_contract,
        "adapter_contract_sha256": canonical_sha256(adapter_contract),
        "capture_target": capture_target,
        "native_env_hash": runnability["native_env_hash"],
        "invocation": invocation,
        "trace_budget": budget,
        "control_bits": copy.deepcopy(control_bits),
        "runnability_evidence": evidence,
    }
    variants: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if problem_level:
        variant = {
            **base,
            "variant_id": "native",
            "cohort": "native",
            "claim_scope": "native_workload_row",
            "native_predictive": True,
            "native_shape": True,
            "problem_level_runnable": True,
        }
        cell_contract = {
            "identity": {"roster_source": "sol_execbench", "slug": slug},
            "category": category,
            "flashinfer_corpus": copy.deepcopy(flashinfer_corpus),
            "launch_scope": "all_in_profiler_roi",
            "launch_completeness_contract": copy.deepcopy(
                LAUNCH_COMPLETENESS_CONTRACT
            ),
            "invocation": invocation,
            "trace_budget": budget,
            "control_bits": control_bits,
        }
        variant["supplemental_cell_contract_sha256"] = canonical_sha256(cell_contract)
        variant_sha = canonical_sha256(variant)
        variants.append({**variant, "variant_contract_sha256": variant_sha})
        for source_row, _diagnostic, diagnostic_sha in passed_rows:
            rows.append(_native_row_document(
                variant=variant, variant_sha=variant_sha,
                source_row=source_row, diagnostic_sha=diagnostic_sha,
            ))
    else:
        for source_row, _diagnostic, diagnostic_sha in passed_rows:
            variant_id = partial_native_variant_id(
                slug=slug, uuid=source_row["source_workload_uuid"],
                row_sha256=source_row["source_row_sha256"],
                diagnostic_sha256=diagnostic_sha, cell_sha256=cell_sha,
                resume_identity_sha256=runnability["resume_identity_sha256"],
            )
            partial_evidence = {
                **evidence,
                "selected_source_workload_index": source_row[
                    "source_workload_index"
                ],
                "selected_source_workload_uuid": source_row[
                    "source_workload_uuid"
                ],
                "selected_source_axes": copy.deepcopy(source_row["source_axes"]),
                "selected_source_row_sha256": source_row["source_row_sha256"],
                "selected_diagnostic_sha256": diagnostic_sha,
            }
            partial_invocation = copy.deepcopy(invocation)
            partial_invocation["logical_workload"][
                "evaluation_evidence"
            ] = partial_evidence
            variant = {
                **base,
                "invocation": partial_invocation,
                "runnability_evidence": partial_evidence,
                "variant_id": variant_id,
                "cohort": "partial_native_row",
                "claim_scope": "partial_native_workload_row_only",
                "native_predictive": False,
                "native_shape": True,
                "problem_level_runnable": False,
                "selected_diagnostic_sha256": diagnostic_sha,
            }
            cell_contract = {
                "identity": {"roster_source": "sol_execbench", "slug": slug},
                "category": category,
                "flashinfer_corpus": copy.deepcopy(flashinfer_corpus),
                "launch_scope": "all_in_profiler_roi",
                "launch_completeness_contract": copy.deepcopy(
                    LAUNCH_COMPLETENESS_CONTRACT
                ),
                "invocation": partial_invocation,
                "trace_budget": budget,
                "control_bits": control_bits,
                "selected_diagnostic_sha256": diagnostic_sha,
            }
            variant["supplemental_cell_contract_sha256"] = canonical_sha256(
                cell_contract
            )
            variant_sha = canonical_sha256(variant)
            variants.append({**variant, "variant_contract_sha256": variant_sha})
            rows.append(_native_row_document(
                variant=variant, variant_sha=variant_sha,
                source_row=source_row, diagnostic_sha=diagnostic_sha,
            ))
    return variants, rows


def _native_row_document(
    *, variant: dict[str, Any], variant_sha: str,
    source_row: dict[str, Any], diagnostic_sha: str,
) -> dict[str, Any]:
    binding = {
        key: copy.deepcopy(source_row[key])
        for key in (
            "source_workload_index", "source_workload_uuid",
            "source_axes", "source_row_sha256",
        )
    }
    return {
        "key": {
            "slug": variant["slug"],
            "source_workload_uuid": source_row["source_workload_uuid"],
            "variant_id": variant["variant_id"],
        },
        "cohort": variant["cohort"],
        "category": variant["category"],
        "flashinfer_corpus": copy.deepcopy(
            variant["adapter_contract"]["flashinfer_corpus"]
        ),
        "launch_scope": "all_in_profiler_roi",
        "launch_completeness_contract": copy.deepcopy(
            LAUNCH_COMPLETENESS_CONTRACT
        ),
        "claim_scope": variant["claim_scope"],
        "native_predictive": variant["native_predictive"],
        "native_shape": True,
        "problem_level_runnable": variant["problem_level_runnable"],
        "variant_contract_sha256": variant_sha,
        "row_evidence_sha256": diagnostic_sha,
        **binding,
        "capture_workload_index": source_row["source_workload_index"],
        "capture_axes": copy.deepcopy(source_row["source_axes"]),
        "capture_row_sha256": source_row["source_row_sha256"],
    }


def _workload_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().split(b"\n")
    except OSError as exc:
        raise WorkloadMatrixError(f"cannot read workload file {path}: {exc}") from exc
    return _parse_workload_lines(lines, label=str(path))


def _workload_rows_bytes(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    """Parse exact bytes already read through an external-owner dirfd."""
    return _parse_workload_lines(payload.split(b"\n"), label=label)


def _parse_workload_lines(
    lines: list[bytes], *, label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    for physical_line, raw_line in enumerate(lines, start=1):
        parse_bytes = raw_line.strip()
        if not parse_bytes or parse_bytes.startswith(b"//"):
            continue
        source_index = len(rows)
        try:
            value = json.loads(parse_bytes)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WorkloadMatrixError(
                f"invalid workload JSON {label}:{physical_line}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise WorkloadMatrixError(
                f"workload row is not an object: {label}:{physical_line}"
            )
        uuid = value.get("uuid")
        axes = value.get("axes")
        if not isinstance(uuid, str) or UUID_RE.fullmatch(uuid) is None:
            raise WorkloadMatrixError(
                f"workload row lacks UUID: {label}:{physical_line}"
            )
        if uuid in seen_uuids:
            raise WorkloadMatrixError(f"duplicate workload UUID {uuid!r}: {label}")
        if (
            not isinstance(axes, dict)
            or any(not isinstance(key, str) or not key for key in axes)
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in axes.values())
        ):
            raise WorkloadMatrixError(
                f"workload axes are not integer-valued: {label}:{physical_line}"
            )
        seen_uuids.add(uuid)
        rows.append({
            "source_workload_index": source_index,
            "source_workload_uuid": uuid,
            "source_axes": copy.deepcopy(axes),
            "source_row_sha256": sha256_bytes(raw_line),
            "source_line": physical_line,
            "source_value": copy.deepcopy(value),
        })
    if not rows:
        raise WorkloadMatrixError(f"workload file contains no rows: {label}")
    return rows


def _axes_delta_matches(
    delta: Any, native_axes: dict[str, Any], reduced_axes: dict[str, Any],
) -> bool:
    if not isinstance(delta, list) or set(native_axes) != set(reduced_axes):
        return False
    expected = []
    for axis in sorted(native_axes):
        if native_axes[axis] == reduced_axes[axis]:
            continue
        source = native_axes[axis]
        reduced = reduced_axes[axis]
        expected.append({
            "axis": axis,
            "from": source,
            "to": reduced,
            "factor": (
                source / reduced
                if isinstance(source, (int, float))
                and not isinstance(source, bool)
                and isinstance(reduced, (int, float))
                and not isinstance(reduced, bool)
                and reduced
                else None
            ),
        })
    return delta == expected


def _json_pointer_segment(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _definition_delta(
    native: Any, reduced: Any,
) -> list[dict[str, Any]] | None:
    """Return the exact leaf-only RFC 6901 delta, or None for shape drift."""
    changes: list[dict[str, Any]] = []

    def walk(source: Any, target: Any, pointer: str) -> bool:
        if isinstance(source, dict) or isinstance(target, dict):
            if not isinstance(source, dict) or not isinstance(target, dict):
                return False
            if set(source) != set(target):
                return False
            for key in sorted(source):
                if not isinstance(key, str):
                    return False
                child = f"{pointer}/{_json_pointer_segment(key)}"
                if not walk(source[key], target[key], child):
                    return False
            return True
        if isinstance(source, list) or isinstance(target, list):
            if (
                not isinstance(source, list) or not isinstance(target, list)
                or len(source) != len(target)
            ):
                return False
            for index, (source_item, target_item) in enumerate(
                zip(source, target)
            ):
                if not walk(source_item, target_item, f"{pointer}/{index}"):
                    return False
            return True
        if type(source) is not type(target) or source != target:
            changes.append({
                "json_pointer": pointer,
                "from": copy.deepcopy(source),
                "to": copy.deepcopy(target),
            })
        return True

    if not walk(native, reduced, ""):
        return None
    return changes


def _definition_delta_matches(
    delta: Any, native: dict[str, Any], reduced: dict[str, Any],
) -> bool:
    if not isinstance(delta, list):
        return False
    expected = _definition_delta(native, reduced)
    return expected is not None and delta == expected


def _reduced_variants(
    *, repo_root: Path, provenance: dict[str, Any], provenance_path: Path,
    provenance_sha256: str, reduced_output_root: Path,
    roster: dict[str, dict[str, Any]], policy: dict[str, Any],
    explicit_budgets: dict[str, dict[str, Any]], control_bits: dict[str, Any],
    flashinfer_contracts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if provenance.get("provenance_contract_version") != 3:
        raise WorkloadMatrixError(
            "unsupported SOL shrink provenance contract: "
            f"{provenance.get('provenance_contract_version')!r}; expected 3"
        )
    try:
        reduced_root_identity = shrink.validate_external_reduced_output_root(
            reduced_output_root
        )
    except sr.PrerequisiteContractError as exc:
        raise WorkloadMatrixError(
            f"invalid reduced-output owner root: {exc}"
        ) from exc
    cells = provenance.get("cells")
    if not isinstance(cells, list):
        raise WorkloadMatrixError("v3 shrink provenance cells are absent")
    provenance_sha = provenance_sha256
    variants: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise WorkloadMatrixError("v3 shrink provenance cell is malformed")
        slug = cell.get("slug")
        if not isinstance(slug, str) or not slug or slug in seen_cells:
            raise WorkloadMatrixError("v3 shrink provenance slug is invalid")
        seen_cells.add(slug)
        eligible = (
            cell.get("provenance_contract_version") == 3
            and cell.get("classification") == "reduced_representative"
            and cell.get("claim_scope") == "reduced_shape_only"
            and cell.get("native_predictive") is False
            and cell.get("reduced_run_status") == "runnable"
            and cell.get("correctness_result") == "passed"
        )
        if not eligible:
            continue
        roster_row = roster.get(slug)
        if not isinstance(roster_row, dict):
            raise WorkloadMatrixError(
                f"accepted geometry proxy is outside the current SOL roster: {slug}"
            )
        category = roster_row["category"]
        language = roster_row["language"]
        cohort_ref = cell.get("reduced_workload_ref")
        if (
            not isinstance(cohort_ref, str) or not cohort_ref
            or Path(cohort_ref).name != cohort_ref
        ):
            raise WorkloadMatrixError(
                f"geometry-proxy cohort owner-relative name invalid: {slug}"
            )
        try:
            cohort_path = shrink._external_output_path(
                reduced_root_identity, cohort_ref
            )
            cohort_payload = shrink._read_bound_direct_child(
                reduced_root_identity, cohort_ref
            )
        except (OSError, sr.PrerequisiteContractError) as exc:
            raise WorkloadMatrixError(
                f"geometry-proxy cohort owner binding invalid: {slug}: {exc}"
            ) from exc
        cohort_rel = cohort_ref
        cohort_sha256 = sha256_bytes(cohort_payload)
        if cohort_sha256 != cell.get("reduced_workload_sha256"):
            raise WorkloadMatrixError(f"geometry-proxy cohort binding mismatch: {slug}")
        external_workload_source = {
            "kind": "shrink_external_child_v1",
            "name": cohort_ref,
            "sha256": cohort_sha256,
            "size_bytes": len(cohort_payload),
            "owner_marker": copy.deepcopy(provenance["reduced_output_owner"]),
            "runtime_rebind": "campaign_retained_immutable_child",
        }
        cohort_rows = _workload_rows_bytes(
            cohort_payload, label=f"external:{cohort_ref}"
        )
        native_path, native_rel = _repo_file(
            repo_root, cell.get("native_shape_ref"),
            label=f"{slug} native workload",
        )
        if sha256_file(native_path) != cell.get("native_workload_sha256"):
            raise WorkloadMatrixError(f"geometry-proxy native workload drift: {slug}")
        native_rows = _workload_rows(native_path)
        if cell.get("source_row_count") != len(native_rows):
            raise WorkloadMatrixError(f"geometry-proxy source row count mismatch: {slug}")
        solution_rel = (
            SOL_BASELINES_ROOT / category / slug / language / "solution.json"
        )
        solution_path, solution_ref = _repo_file(
            repo_root, solution_rel.as_posix(), label=f"{slug} solution"
        )
        solution = _load_object(solution_path, label=f"{slug} solution")
        definition_name = solution.get("definition")
        if not isinstance(definition_name, str) or not definition_name:
            raise WorkloadMatrixError(f"geometry-proxy solution is invalid: {slug}")
        benchmark_rel = (
            SOL_BENCHMARK_ROOT / SUBSET_BY_CATEGORY[category] / definition_name
        )
        definition_path, definition_rel = _repo_file(
            repo_root, (benchmark_rel / "definition.json").as_posix(),
            label=f"{slug} definition",
        )
        native_definition = _load_object(
            definition_path, label=f"{slug} native definition"
        )
        native_definition_sha256 = sha256_file(definition_path)
        if (
            cell.get("definition_ref") != definition_rel
            or cell.get("definition_sha256") != native_definition_sha256
            or native_rel != (benchmark_rel / "workload.jsonl").as_posix()
        ):
            raise WorkloadMatrixError(
                f"geometry-proxy definition/native source binding mismatch: {slug}"
            )
        reduced_definition_ref = cell.get("reduced_definition_ref")
        reduced_definition_sha256 = cell.get("reduced_definition_sha256")
        definition_delta = cell.get("definition_delta")
        external_definition_source: dict[str, Any] | None
        if reduced_definition_ref is None:
            if reduced_definition_sha256 is not None or definition_delta != []:
                raise WorkloadMatrixError(
                    f"geometry-proxy unchanged definition binding invalid: {slug}"
                )
            effective_definition_ref = definition_rel
            effective_definition_sha256 = native_definition_sha256
            effective_definition = native_definition
            external_definition_source = None
        else:
            if (
                not isinstance(reduced_definition_ref, str)
                or not reduced_definition_ref
                or Path(reduced_definition_ref).name != reduced_definition_ref
                or SHA256_RE.fullmatch(str(reduced_definition_sha256)) is None
                or not isinstance(definition_delta, list)
                or not definition_delta
            ):
                raise WorkloadMatrixError(
                    f"geometry-proxy reduced definition binding invalid: {slug}"
                )
            try:
                reduced_definition_payload = shrink._read_bound_direct_child(
                    reduced_root_identity, reduced_definition_ref
                )
            except (OSError, sr.PrerequisiteContractError) as exc:
                raise WorkloadMatrixError(
                    f"geometry-proxy reduced definition owner binding invalid: "
                    f"{slug}: {exc}"
                ) from exc
            observed_reduced_definition_sha256 = sha256_bytes(
                reduced_definition_payload
            )
            if observed_reduced_definition_sha256 != reduced_definition_sha256:
                raise WorkloadMatrixError(
                    f"geometry-proxy reduced definition binding mismatch: {slug}"
                )
            try:
                effective_definition = json.loads(reduced_definition_payload)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise WorkloadMatrixError(
                    f"geometry-proxy reduced definition is invalid: {slug}: {exc}"
                ) from exc
            if (
                not isinstance(effective_definition, dict)
                or effective_definition.get("name") != definition_name
                or native_definition.get("name") != definition_name
                or observed_reduced_definition_sha256
                == native_definition_sha256
                or not _definition_delta_matches(
                    definition_delta, native_definition, effective_definition
                )
            ):
                raise WorkloadMatrixError(
                    f"geometry-proxy reduced definition contract mismatch: {slug}"
                )
            effective_definition_ref = reduced_definition_ref
            effective_definition_sha256 = observed_reduced_definition_sha256
            external_definition_source = {
                "kind": "shrink_external_definition_child_v1",
                "name": reduced_definition_ref,
                "sha256": observed_reduced_definition_sha256,
                "size_bytes": len(reduced_definition_payload),
                "owner_marker": copy.deepcopy(provenance["reduced_output_owner"]),
                "runtime_rebind": "campaign_retained_immutable_child",
            }
        cohort_evidence = cell.get("cohort_execution_evidence")
        if not isinstance(cohort_evidence, dict):
            raise WorkloadMatrixError(
                f"geometry-proxy cohort execution evidence is absent: {slug}"
            )
        expected_inputs = {
            "definition_input": {
                "path": effective_definition_ref,
                "sha256": effective_definition_sha256,
            },
            "workload_input": {
                "path": cohort_rel, "sha256": cohort_sha256,
            },
            "solution_input": {
                "path": solution_ref, "sha256": sha256_file(solution_path),
            },
        }
        if any(
            cohort_evidence.get(field) != expected
            for field, expected in expected_inputs.items()
        ):
            raise WorkloadMatrixError(
                f"geometry-proxy execution input binding mismatch: {slug}"
            )
        executable_path, executable_ref = _repo_file(
            repo_root, SOL_EXECUTABLE_PATH.as_posix(),
            label="SOL adapter executable",
        )
        forwarded = [
            "--definition", effective_definition_ref,
            "--workload", cohort_rel,
            "--solution", solution_ref,
            "--json",
        ]
        flashinfer_corpus = flashinfer_contracts.get(category)
        if not isinstance(flashinfer_corpus, dict):
            raise WorkloadMatrixError(
                f"SOL FlashInfer corpus contract is absent: {slug}:{category}"
            )
        adapter_contract = {
            "contract_version": 3,
            "forwarded_argv": forwarded,
            "environment": copy.deepcopy(EXACT_SOL_ENVIRONMENT),
            "native_env_hash": provenance["native_env_hash"],
            "flashinfer_corpus": copy.deepcopy(flashinfer_corpus),
            "shape_forwarded_to_target": False,
            "label_driver_bridge_status": "implemented",
            "persistent_stage_status": "implemented",
            "capture_request_environment": "SOL_EXECBENCH_CAPTURE_REQUEST",
            "label_driver_bridge_rule": (
                "forward adapter_contract.forwarded_argv exactly without "
                "appending a synthetic --shape-point token"
            ),
        }
        capture_target = copy.deepcopy(SUPPLEMENTAL_CAPTURE_TARGET)
        evaluation_evidence = {
            "source_kind": "current_reduced_proxy_execution",
            "source_path": provenance_path.name,
            "source_sha256": provenance_sha,
            "cell_sha256": canonical_sha256(cell),
            "cohort_execution_evidence_sha256": canonical_sha256(
                cohort_evidence
            ),
            "native_env_hash": provenance["native_env_hash"],
            "status": "runnable",
            "returncode": 0,
            "expected_workloads": len(cohort_rows),
            "detail": cohort_evidence.get("detail"),
        }
        invocation = {
            "adapter_kind": "sol_execbench",
            "authority_reference_path": solution_ref,
            "authority_reference_sha256": sha256_file(solution_path),
            "executable_path": executable_ref,
            "executable_sha256": sha256_file(executable_path),
            "label_app_args": forwarded,
            "trace_app_args": forwarded,
            "evaluation_cardinality_contract": (
                "all_effective_workloads_passed_exact"
            ),
            "trace_cardinality_contract": SUPPLEMENTAL_TRACE_CARDINALITY_CONTRACT,
            "capture_launch_contract": copy.deepcopy(
                SUPPLEMENTAL_CAPTURE_LAUNCH_CONTRACT
            ),
            "capture_target": capture_target,
            "logical_workload": {
                "kind": "sol_execbench_workload_set",
                "value": "reduced_representative",
                "workload_scope": "reduced_representative",
                "benchmark_definition": definition_name,
                "definition_file": effective_definition_ref,
                "definition_file_sha256": effective_definition_sha256,
                "native_definition_file": definition_rel,
                "native_definition_file_sha256": native_definition_sha256,
                "external_definition_source": copy.deepcopy(
                    external_definition_source
                ),
                "benchmark_dir": benchmark_rel.as_posix(),
                "benchmark_tree_sha256": tree_sha256(
                    repo_root / benchmark_rel
                ),
                "workload_file": cohort_rel,
                "workload_file_sha256": cohort_sha256,
                "external_workload_source": copy.deepcopy(
                    external_workload_source
                ),
                "expected_workloads": len(cohort_rows),
                "evaluation_evidence": evaluation_evidence,
            },
            "shape_delivery": "adapter_consumes_logical_shape",
            "target_processes": "all",
            "adapter_contract": adapter_contract,
        }
        budget = _trace_budget(
            slug, policy=policy, explicit=explicit_budgets
        )
        base_variant = {
            "slug": slug,
            "category": category,
            "launch_scope": "all_in_profiler_roi",
            "launch_completeness_contract": copy.deepcopy(
                LAUNCH_COMPLETENESS_CONTRACT
            ),
            "workload_scope": "reduced_representative",
            "source_workload_file": cohort_rel,
            "source_workload_file_sha256": cohort_sha256,
            "source_workload_count": len(cohort_rows),
            "definition_file": effective_definition_ref,
            "definition_file_sha256": effective_definition_sha256,
            "solution_file": solution_ref,
            "solution_file_sha256": sha256_file(solution_path),
            "adapter_contract": adapter_contract,
            "adapter_contract_sha256": canonical_sha256(adapter_contract),
            "capture_target": capture_target,
            "native_env_hash": provenance["native_env_hash"],
            "invocation": invocation,
            "trace_budget": budget,
            "control_bits": copy.deepcopy(control_bits),
        }
        plans = cell.get("row_plans")
        if not isinstance(plans, list):
            raise WorkloadMatrixError(f"geometry-proxy row plans are absent: {slug}")
        planned = [
            plan for plan in plans
            if isinstance(plan, dict) and plan.get("planning_status") == "proxy_planned"
        ]
        if not planned or len(planned) != len(cohort_rows):
            raise WorkloadMatrixError(f"geometry-proxy cohort is not bijective: {slug}")
        planned_by_uuid: dict[str, dict[str, Any]] = {}
        for plan in planned:
            uuid = plan.get("source_workload_uuid")
            if not isinstance(uuid, str) or uuid in planned_by_uuid:
                raise WorkloadMatrixError(f"geometry-proxy plan UUID is invalid: {slug}")
            planned_by_uuid[uuid] = plan
        cohort_by_uuid = {
            row["source_workload_uuid"]: row for row in cohort_rows
        }
        if set(cohort_by_uuid) != set(planned_by_uuid):
            raise WorkloadMatrixError(f"geometry-proxy UUID bijection mismatch: {slug}")
        if len(cohort_by_uuid) != len(cohort_rows):
            raise WorkloadMatrixError(f"geometry-proxy cohort UUID is duplicated: {slug}")
        cell_sha = canonical_sha256(cell)
        for uuid, plan in planned_by_uuid.items():
            source_index = plan.get("source_workload_index")
            if (
                type(source_index) is not int
                or source_index < 0
                or source_index >= len(native_rows)
            ):
                raise WorkloadMatrixError(
                    f"geometry-proxy native index is invalid: {slug}:{uuid}"
                )
            native_row = native_rows[source_index]
            capture_row = cohort_by_uuid[uuid]
            model = plan.get("model")
            model_id = model.get("model_id") if isinstance(model, dict) else None
            variant_id = plan.get("variant_id")
            reduced_sha = plan.get("reduced_row_sha256")
            native_sha = plan.get("native_row_sha256")
            if (
                plan.get("slug") != slug
                or plan.get("claim_scope") != "reduced_shape_only"
                or plan.get("native_predictive") is not False
                or plan.get("reduced_run_status") != "runnable"
                or plan.get("correctness_result") != "passed"
                or plan.get("row_sha256_scope")
                != "exact_json_line_bytes_excluding_lf_only"
                or not isinstance(model_id, str)
                or SHA256_RE.fullmatch(str(variant_id)) is None
                or SHA256_RE.fullmatch(str(native_sha)) is None
                or SHA256_RE.fullmatch(str(reduced_sha)) is None
                or (
                    native_sha == reduced_sha
                    and native_definition_sha256
                    == effective_definition_sha256
                )
                or plan.get("definition_ref") != definition_rel
                or plan.get("definition_sha256")
                != native_definition_sha256
                or plan.get("reduced_definition_ref")
                != reduced_definition_ref
                or plan.get("reduced_definition_sha256")
                != reduced_definition_sha256
                or plan.get("definition_delta") != definition_delta
                or native_row["source_workload_uuid"] != uuid
                or native_row["source_row_sha256"] != native_sha
                or native_row["source_axes"] != plan.get("native_axes")
                or capture_row["source_row_sha256"] != reduced_sha
                or capture_row["source_axes"] != plan.get("reduced_axes")
                or not _axes_delta_matches(
                    plan.get("axes_delta"), native_row["source_axes"],
                    capture_row["source_axes"],
                )
                or variant_id != producer_variant_id(
                    slug=slug, source_uuid=uuid, native_sha256=native_sha,
                    reduced_sha256=reduced_sha,
                    native_definition_sha256=native_definition_sha256,
                    reduced_definition_sha256=reduced_definition_sha256,
                    model_id=model_id,
                )
            ):
                raise WorkloadMatrixError(
                    f"geometry-proxy row binding mismatch: {slug}:{uuid}"
                )
            variant = copy.deepcopy(base_variant)
            variant.update({
                "variant_id": variant_id,
                "cohort": "reduced_proxy",
                "claim_scope": REDUCED_PROXY_CLAIM_SCOPE,
                "native_predictive": False,
                "native_shape": False,
                "problem_level_runnable": False,
                "source_workload_file": cohort_rel,
                "source_workload_file_sha256": cohort_sha256,
                "source_workload_count": len(cohort_rows),
                "reduced_provenance": {
                    "source_path": provenance_path.name,
                    "source_sha256": provenance_sha,
                    "provenance_contract_version": 3,
                    "cell_sha256": cell_sha,
                    "cohort_id": cell.get("cohort_id"),
                    "native_workload_file": native_rel,
                    "native_workload_file_sha256": cell.get("native_workload_sha256"),
                    "native_definition_file": definition_rel,
                    "native_definition_file_sha256": native_definition_sha256,
                    "reduced_definition_file": effective_definition_ref,
                    "reduced_definition_file_sha256": (
                        effective_definition_sha256
                    ),
                    "definition_delta": copy.deepcopy(definition_delta),
                    "row_plan_sha256": canonical_sha256(plan),
                    "axes_delta": copy.deepcopy(plan["axes_delta"]),
                },
            })
            supplemental_cell = {
                "identity": {"roster_source": "sol_execbench", "slug": slug},
                "category": category,
                "flashinfer_corpus": copy.deepcopy(flashinfer_corpus),
                "launch_scope": "all_in_profiler_roi",
                "launch_completeness_contract": copy.deepcopy(
                    LAUNCH_COMPLETENESS_CONTRACT
                ),
                "invocation": variant["invocation"],
                "trace_budget": variant["trace_budget"],
                "control_bits": variant["control_bits"],
                "row_plan_sha256": variant["reduced_provenance"][
                    "row_plan_sha256"
                ],
            }
            variant["supplemental_cell_contract_sha256"] = canonical_sha256(
                supplemental_cell
            )
            variant_contract_sha256 = canonical_sha256(variant)
            variants.append({
                **variant,
                "variant_contract_sha256": variant_contract_sha256,
            })
            rows.append({
                "key": {
                    "slug": slug,
                    "source_workload_uuid": uuid,
                    "variant_id": variant_id,
                },
                "cohort": "reduced_proxy",
                "category": category,
                "flashinfer_corpus": copy.deepcopy(flashinfer_corpus),
                "launch_scope": "all_in_profiler_roi",
                "launch_completeness_contract": copy.deepcopy(
                    LAUNCH_COMPLETENESS_CONTRACT
                ),
                "claim_scope": REDUCED_PROXY_CLAIM_SCOPE,
                "native_predictive": False,
                "native_shape": False,
                "problem_level_runnable": False,
                "variant_contract_sha256": variant_contract_sha256,
                "row_evidence_sha256": canonical_sha256(plan),
                "source_workload_index": source_index,
                "source_workload_uuid": uuid,
                "source_axes": copy.deepcopy(native_row["source_axes"]),
                "source_row_sha256": native_sha,
                "capture_workload_index": capture_row["source_workload_index"],
                "capture_axes": copy.deepcopy(capture_row["source_axes"]),
                "capture_row_sha256": reduced_sha,
            })
    return variants, rows


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    key = row["key"]
    return (
        COHORT_ORDER[row["cohort"]], key["slug"],
        key["source_workload_uuid"], key["variant_id"],
    )


def _chunks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    schedules: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cohort in ("native", "partial_native_row", "reduced_proxy"):
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        by_variant: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in cohort_rows:
            key = row["key"]
            by_variant.setdefault(
                (key["slug"], key["variant_id"]), []
            ).append(row)
        waves: list[list[dict[str, Any]]] = []
        depth = 0
        while True:
            wave = [
                members[depth]
                for _identity, members in sorted(by_variant.items())
                if depth < len(members)
            ]
            if not wave:
                break
            waves.append(wave)
            depth += 1
        scheduled = [row for wave in waves for row in wave]
        if len(scheduled) != len(cohort_rows):
            raise WorkloadMatrixError("breadth-first chunk scheduling lost rows")
        schedules[cohort] = {
            "representative": waves[0] if waves else [],
            "depth": [row for wave in waves[1:] for row in wave],
        }
    ordinals = {cohort: 0 for cohort in schedules}
    for tier in ("representative", "depth"):
        tier_chunks: dict[str, list[dict[str, Any]]] = {
            cohort: [] for cohort in schedules
        }
        for cohort in ("native", "partial_native_row", "reduced_proxy"):
            scheduled = schedules[cohort][tier]
            for offset in range(0, len(scheduled), ROWS_PER_CHUNK):
                members = [copy.deepcopy(row["key"]) for row in scheduled[
                    offset:offset + ROWS_PER_CHUNK
                ]]
                ordinals[cohort] += 1
                ordinal = ordinals[cohort]
                tier_chunks[cohort].append({
                    "chunk_id": f"sol-workload-{cohort.replace('_', '-')}-{ordinal:03d}",
                    "cohort": cohort,
                    "tier": tier,
                    "ordinal": ordinal,
                    "row_count": len(members),
                    "membership_sha256": canonical_sha256(members),
                    "rows": members,
                })
        # Cohort-pure chunks are interleaved by wave so no large cohort can
        # consume its full representative tier before the others are sampled.
        for wave in range(max((len(items) for items in tier_chunks.values()), default=0)):
            for cohort in ("native", "partial_native_row", "reduced_proxy"):
                items = tier_chunks[cohort]
                if wave < len(items):
                    result.append(items[wave])
    return result


def _live_source_commit(
    runnability: dict[str, Any], provenance: dict[str, Any],
) -> str:
    """Bind the live supplemental producer, not the frozen base matrix.

    The canonical 344-cell matrix retains its historical source commit as
    provenance.  Supplemental runnability, shrink, planning, and execution are
    produced from one current clean commit and must agree with the checkout in
    which this matrix is generated.
    """
    current = sr.source_head()
    sr.source_worktree_state()
    run_commit = runnability.get("source_commit")
    shrink_commit = provenance.get("source_commit")
    if (
        re.fullmatch(r"[0-9a-f]{40}", current) is None
        or run_commit != current
        or shrink_commit != current
    ):
        raise WorkloadMatrixError(
            "live runnability, shrink, and producer source commits differ: "
            f"HEAD={current}; runnability={run_commit}; shrink={shrink_commit}"
        )
    return current


def build_matrix(
    repo_root: Path = REPO_ROOT,
    *,
    canonical_matrix_path: Path | None = None,
    authority_path: Path | None = None,
    runnability_path: Path | None = None,
    shrink_path: Path | None = None,
    runnability_root: Path | None = None,
    shrink_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve()
    live_native_env_pin = _live_native_env_pin(root)
    source_path = _explicit_input_file(
        root, canonical_matrix_path, CANONICAL_MATRIX_PATH,
        label="canonical capture matrix",
    )
    canonical = _load_object(source_path, label="canonical capture matrix")
    if (
        canonical.get("schema_version") != 1
        or canonical.get("arch") != "sm_86"
        or canonical.get("sku") != "rtx_3090_ti"
        or not SHA256_RE.fullmatch(str(canonical.get("matrix_seal_sha256", "")))
    ):
        raise WorkloadMatrixError("canonical capture matrix identity is invalid")
    unsealed = {
        key: value for key, value in canonical.items()
        if key != "matrix_seal_sha256"
    }
    if canonical_source_sha256(unsealed) != canonical["matrix_seal_sha256"]:
        raise WorkloadMatrixError("canonical capture matrix seal mismatch")
    authority = canonical.get("authority")
    if not isinstance(authority, dict):
        raise WorkloadMatrixError("canonical capture authority pin is absent")
    recorded_authority_path, _recorded_authority_ref = _repo_file(
        root, authority.get("path"), label="canonical capture authority",
        allow_external=True,
    )
    resolved_authority_path = _explicit_input_file(
        root, authority_path, recorded_authority_path,
        label="canonical capture authority override",
    )
    if resolved_authority_path != recorded_authority_path:
        raise WorkloadMatrixError(
            "explicit authority does not equal the path sealed by canonical matrix"
        )
    authority_rel = _display_path(resolved_authority_path, root)
    if sha256_file(resolved_authority_path) != authority.get("sha256"):
        raise WorkloadMatrixError("canonical capture authority hash mismatch")
    _load_object(resolved_authority_path, label="canonical capture authority")
    cells = canonical.get("cells")
    if not isinstance(cells, list):
        raise WorkloadMatrixError("canonical capture matrix cells are absent")
    rows: list[dict[str, Any]] = []
    variant_documents: list[dict[str, Any]] = []
    roster, roster_sha = _load_roster(root)
    runnability_path = _explicit_input_file(
        root, runnability_path, RUNNABILITY_PATH,
        label="completed SOL runnability authority",
    )
    if runnability_root is None or shrink_root is None:
        raise WorkloadMatrixError(
            "live supplemental matrix requires explicit runnability and shrink owner roots"
        )
    runnability_root = Path(runnability_root)
    shrink_root = Path(shrink_root)
    runnability, runnability_sha, probe_contract = _load_current_runnability(
        runnability_path, repo_root=root, roster=roster,
        external_campaign_root=runnability_root,
    )
    if runnability.get("native_env_hash") != live_native_env_pin["native_env_hash"]:
        raise WorkloadMatrixError(
            "SOL runnability authority does not match the live fingerprint"
        )
    flashinfer_contracts = _flashinfer_contracts(probe_contract)
    policy, explicit_budgets = _load_budget_policy(root)
    control_bits = _control_contract()
    runnability_cells = {
        cell["slug"]: cell for cell in runnability["cells"]
    }
    for slug in sorted(roster):
        variants, native_rows = _native_problem_contract(
            repo_root=root, record=runnability_cells[slug],
            roster_row=roster[slug], runnability=runnability,
            runnability_path=runnability_path, policy=policy,
            explicit_budgets=explicit_budgets, control_bits=control_bits,
            flashinfer_contracts=flashinfer_contracts,
        )
        variant_documents.extend(variants)
        rows.extend(native_rows)
    provenance_path = _explicit_input_file(
        root, shrink_path, SHRINK_PROVENANCE_PATH,
        label="completed SOL shrink provenance",
    )
    try:
        provenance, provenance_sha = shrink.load_completed_shrink_provenance(
            provenance_path,
            runnability=runnability,
            runnability_sha256=runnability_sha,
            probe_contract=probe_contract,
            targets=shrink.host_capacity_slugs(runnability),
            runnability_path=runnability_path,
            reduced_output_root=shrink_root,
        )
    except (
        OSError, ValueError, sr.PrerequisiteContractError,
        jsonschema.ValidationError,
    ) as exc:
        raise WorkloadMatrixError(
            f"SOL geometry-proxy authority is not complete/current: {exc}"
        ) from exc
    if provenance.get("native_env_hash") != live_native_env_pin["native_env_hash"]:
        raise WorkloadMatrixError(
            "SOL geometry-proxy authority does not match the live fingerprint"
        )
    reduced_variants, reduced_rows = _reduced_variants(
        repo_root=root, provenance=provenance,
        provenance_path=provenance_path,
        provenance_sha256=provenance_sha,
        roster=roster, policy=policy,
        explicit_budgets=explicit_budgets, control_bits=control_bits,
        flashinfer_contracts=flashinfer_contracts,
        reduced_output_root=shrink_root,
    )
    variant_documents.extend(reduced_variants)
    rows.extend(reduced_rows)
    live_source_commit = _live_source_commit(runnability, provenance)
    variant_keys = [
        (item["slug"], item["variant_id"]) for item in variant_documents
    ]
    if len(variant_keys) != len(set(variant_keys)):
        raise WorkloadMatrixError("duplicate SOL supplemental variant")
    variant_documents.sort(key=lambda item: (
        COHORT_ORDER[item["cohort"]], item["slug"], item["variant_id"]
    ))
    _validate_variant_native_env_closure(
        variant_documents,
        expected_hash=live_native_env_pin["native_env_hash"],
    )
    rows.sort(key=_row_sort_key)
    keys = [
        (
            row["key"]["slug"], row["key"]["source_workload_uuid"],
            row["key"]["variant_id"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise WorkloadMatrixError("supplemental workload-row key is not unique")
    chunks = _chunks(rows)
    native_rows = sum(row["cohort"] == "native" for row in rows)
    partial_rows = sum(row["cohort"] == "partial_native_row" for row in rows)
    reduced_rows = sum(row["cohort"] == "reduced_proxy" for row in rows)
    native_variants = sum(item["cohort"] == "native" for item in variant_documents)
    partial_variants = sum(
        item["cohort"] == "partial_native_row" for item in variant_documents
    )
    reduced_variants = sum(
        item["cohort"] == "reduced_proxy" for item in variant_documents
    )
    document: dict[str, Any] = {
        "schema_version": 2,
        "artifact_id": "sm86_sol_workload_capture_matrix_live_v2",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "source_commit": live_source_commit,
        "input_pins": {
            "sol_native_env_fingerprint": live_native_env_pin,
        },
        "canonical_matrix": {
            "path": _display_path(source_path, root),
            "sha256": sha256_file(source_path),
            "source_commit": canonical.get("source_commit"),
            "matrix_seal_sha256": canonical["matrix_seal_sha256"],
        },
        "canonical_authority": {
            "path": authority_rel,
            "sha256": authority["sha256"],
            "roster_id": authority.get("roster_id"),
        },
        "runnability_authority": {
            "path": runnability["external_output_name"],
            "sha256": runnability_sha,
            "owner_marker": copy.deepcopy(
                runnability["external_campaign_owner"]
            ),
            "roster_path": ROSTER_PATH.as_posix(),
            "roster_sha256": roster_sha,
            "roster_rows": len(roster),
            "native_env_hash": runnability["native_env_hash"],
            "probe_protocol_sha256": runnability["probe_protocol_sha256"],
            "probe_config_sha256": runnability["probe_config_sha256"],
            "sol_execbench_package_sha256": runnability[
                "sol_execbench_package_sha256"
            ],
            "data_annex_manifest_sha256": runnability[
                "data_annex_manifest_sha256"
            ],
            "flashinfer_corpus_manifest_sha256": runnability[
                "flashinfer_corpus_manifest_sha256"
            ],
            "flashinfer_corpus_file_count": runnability[
                "flashinfer_corpus_file_count"
            ],
            "flashinfer_corpus_blob_count": runnability[
                "flashinfer_corpus_blob_count"
            ],
            "flashinfer_corpus_verified": runnability[
                "flashinfer_corpus_verified"
            ],
            "resume_identity_sha256": runnability["resume_identity_sha256"],
        },
        "supplemental_budget_policy": policy,
        "reduced_proxy_authority": {
            "path": provenance["provenance_output_name"],
            "sha256": provenance_sha,
            "owner_marker": copy.deepcopy(
                provenance["reduced_output_owner"]
            ),
            "provenance_contract_version": provenance.get(
                "provenance_contract_version"
            ),
        },
        "claim_policy": {
            "native_cohort": "native_workload_rows_only",
            "partial_native_cohort": (
                "passed_native_rows_only_no_problem_level_runnability_claim"
            ),
            "reduced_proxy_cohort": REDUCED_PROXY_COHORT_POLICY,
            "cross_cohort_aggregation": "forbidden_without_explicit_stratification",
            "canonical_344_cell_matrix_impact": "none_additive_supplement_only",
        },
        "chunk_policy": {
            "rows_per_chunk": ROWS_PER_CHUNK,
            "cohorts_may_mix": False,
            "ordering": "representatives_across_all_cohorts_before_depth",
            "tier_order": ["representative", "depth"],
        },
        "counts": {
            "problems": len({item[0] for item in variant_keys}),
            "variants": len(variant_keys),
            "native_variants": native_variants,
            "partial_native_variants": partial_variants,
            "reduced_variants": reduced_variants,
            "rows": len(rows),
            "native_rows": native_rows,
            "partial_native_rows": partial_rows,
            "reduced_proxy_rows": reduced_rows,
            "chunks": len(chunks),
        },
        "variants": variant_documents,
        "rows": rows,
        "chunks": chunks,
    }
    document["matrix_seal_sha256"] = canonical_sha256(document)
    return document


def _deliver_matrix(
    document: dict[str, Any], *, emit: Path | None = None,
    check: Path | None = None,
) -> int:
    """Emit or byte-check one already constructed complete matrix."""
    if (emit is None) == (check is None):
        raise WorkloadMatrixError("exactly one matrix delivery mode is required")
    try:
        schema = json.loads((REPO_ROOT / SCHEMA_PATH).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(document, schema)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError,
            jsonschema.SchemaError) as exc:
        raise WorkloadMatrixError(
            f"supplemental matrix producer output violates schema: {exc}"
        ) from exc
    expected = serialize_matrix(document)
    target = check if check is not None else emit
    assert target is not None
    root_identity = sr.validate_external_campaign_root(target.parent)
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"duplicate plan-owner key: {key!r}")
            parsed[key] = value
        return parsed
    try:
        marker_payload = sr.read_external_child(
            root_identity, LIVE_PLAN_OWNER_FILENAME
        )
        marker = json.loads(
            marker_payload, object_pairs_hook=reject_duplicate_keys
        )
    except (
        json.JSONDecodeError, ValueError, sr.PrerequisiteContractError,
    ) as exc:
        raise WorkloadMatrixError(
            f"live plan owner marker is invalid: {exc}"
        ) from exc
    unsigned_marker = {
        key: value for key, value in marker.items()
        if key != "owner_seal_sha256"
    } if isinstance(marker, dict) else {}
    marker_source_commit = marker.get("source_commit") if isinstance(marker, dict) else None
    marker_matrix_name = marker.get("matrix_name") if isinstance(marker, dict) else None
    marker_inventory_name = marker.get("inventory_name") if isinstance(marker, dict) else None
    safe_matrix_name = (
        isinstance(marker_matrix_name, str)
        and marker_matrix_name not in {"", ".", ".."}
        and "/" not in marker_matrix_name
        and "\\" not in marker_matrix_name
        and Path(marker_matrix_name).name == marker_matrix_name
    )
    safe_inventory_name = (
        isinstance(marker_inventory_name, str)
        and marker_inventory_name not in {"", ".", ".."}
        and "/" not in marker_inventory_name
        and "\\" not in marker_inventory_name
        and Path(marker_inventory_name).name == marker_inventory_name
    )
    if (
        not isinstance(marker, dict)
        or set(marker) != {
            "schema_version", "kind", "source_commit", "matrix_name",
            "inventory_name", "runnability_owner", "shrink_owner",
            "owner_seal_sha256",
        }
        or marker.get("schema_version") != 1
        or marker.get("kind") != LIVE_PLAN_OWNER_KIND
        or not isinstance(marker_source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", marker_source_commit) is None
        or marker.get("source_commit") != document.get("source_commit")
        or not safe_matrix_name
        or not safe_inventory_name
        or marker.get("matrix_name") != target.name
        or marker.get("matrix_name") == marker.get("inventory_name")
        or marker.get("runnability_owner")
        != (document.get("runnability_authority") or {}).get("owner_marker")
        or marker.get("shrink_owner")
        != (document.get("reduced_proxy_authority") or {}).get("owner_marker")
        or marker.get("owner_seal_sha256")
        != sha256_bytes(serialize_matrix(unsigned_marker))
        or marker_payload != serialize_matrix(marker)
    ):
        raise WorkloadMatrixError("live plan owner marker contract mismatch")
    names = sr.external_root_names(root_identity)
    if check is not None:
        if names != {LIVE_PLAN_OWNER_FILENAME, target.name}:
            raise WorkloadMatrixError(
                f"live plan check namespace is not closed: {sorted(names)!r}"
            )
        observed = sr.read_external_child(root_identity, target.name)
        marker_after = sr.read_external_child(
            root_identity, LIVE_PLAN_OWNER_FILENAME
        )
        observed_after = sr.read_external_child(root_identity, target.name)
        names_after = sr.external_root_names(root_identity)
        if (
            marker_after != marker_payload
            or observed_after != observed
            or names_after != names
        ):
            raise WorkloadMatrixError(
                "live plan owner, matrix, or namespace changed during matrix check"
            )
        if observed != expected:
            print(
                f"supplemental SOL workload matrix byte mismatch: {check}",
                file=sys.stderr,
            )
            return 1
        print(
            f"supplemental SOL workload matrix verified: rows={document['counts']['rows']} "
            f"chunks={document['counts']['chunks']}"
        )
        return 0
    assert emit is not None
    if names != {LIVE_PLAN_OWNER_FILENAME}:
        raise WorkloadMatrixError(
            f"live plan emit namespace is not empty: {sorted(names)!r}"
        )
    sr.publish_external_child(root_identity, emit.name, expected)
    names_after = sr.external_root_names(root_identity)
    if names_after != {LIVE_PLAN_OWNER_FILENAME, target.name}:
        raise WorkloadMatrixError(
            f"live plan namespace changed during matrix publication: "
            f"{sorted(names_after)!r}"
        )
    marker_after = sr.read_external_child(
        root_identity, LIVE_PLAN_OWNER_FILENAME
    )
    observed = sr.read_external_child(root_identity, target.name)
    marker_final = sr.read_external_child(
        root_identity, LIVE_PLAN_OWNER_FILENAME
    )
    observed_final = sr.read_external_child(root_identity, target.name)
    names_final = sr.external_root_names(root_identity)
    if marker_after != marker_payload or marker_final != marker_payload:
        raise WorkloadMatrixError(
            "live plan owner changed during matrix publication"
        )
    if observed != expected or observed_final != expected:
        raise WorkloadMatrixError(
            "published supplemental SOL workload matrix byte mismatch"
        )
    if names_final != names_after:
        raise WorkloadMatrixError(
            "live plan namespace changed during matrix publication verification"
        )
    print(
        f"supplemental SOL workload matrix: rows={document['counts']['rows']} "
        f"chunks={document['counts']['chunks']} -> {emit}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", type=Path, help="write canonical matrix bytes")
    mode.add_argument("--check", type=Path, help="byte-check an existing matrix")
    parser.add_argument("--canonical-matrix", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--runnability", type=Path)
    parser.add_argument("--runnability-root", type=Path, required=True)
    parser.add_argument("--shrink-provenance", type=Path)
    parser.add_argument("--shrink-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        document = build_matrix(
            REPO_ROOT,
            canonical_matrix_path=args.canonical_matrix,
            authority_path=args.authority,
            runnability_path=args.runnability,
            shrink_path=args.shrink_provenance,
            runnability_root=args.runnability_root,
            shrink_root=args.shrink_root,
        )
        return _deliver_matrix(document, emit=args.emit, check=args.check)
    except (OSError, WorkloadMatrixError, sr.PrerequisiteContractError) as exc:
        print(f"build_sol_workload_capture_matrix: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
