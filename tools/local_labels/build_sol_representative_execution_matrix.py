#!/usr/bin/env python3
"""Build the clean-HEAD execution authority for the SOL representative census.

The selection authority answers *which* 216 rows represent the 201 sm_86
applications.  This module answers *what exact bytes and execution contract*
may run them.  It deliberately keeps the frozen e387 source authority visible,
proves the selected workload/definition/solution/config bytes across Git trees,
re-pins changed execution tooling, applies a representative-only 900 second
ceiling, and re-seals every selected variant, row, chunk, and the final matrix.

The exhaustive 3,420-row matrix is an immutable input.  It is never rewritten
or treated as the live execution matrix by this builder.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import build_sol_representative_capture_matrix as selector  # noqa: E402
from tools.local_labels import build_sol_workload_capture_matrix as source_builder  # noqa: E402


SCHEMA_PATH = Path(
    "workloads/simbench_bw/schemas/sol_representative_execution_matrix.schema.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/local_labels/campaign/sol_representative_execution_matrix.live.v1.json"
)
ARTIFACT_ID = "sm86_sol_representative_execution_matrix_live_v1"
EXECUTION_PROFILE = "representative_trace_ncu_control_typed_v1"
REPRESENTATIVE_MAX_WALL_TIME_S = 900
REPRESENTATIVE_MAX_TRACE_BYTES = 1024 ** 3
ROWS_PER_CHUNK = 16
SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
SHA40_RE = __import__("re").compile(r"^[0-9a-f]{40}$")
PINNED_AUTHORITY_COMMIT = "e38788ca0a4457570bd562ea1389b07b3f855e22"
PINNED_SOURCE_MATRIX_SHA256 = "8441ac3cedb224cc7c918c91ef11a1add0d920ecd96739f9c8e5b04b815dcde4"
PINNED_SOURCE_MATRIX_SEAL_SHA256 = "e6d213b66d451a8f5e7d095427019434a017a07d272d8e322a8c0fae3ccec7a6"
PINNED_REPRESENTATIVE_SELECTION_SHA256 = "1df582f3a66587106b4e243477c536cfb31f56e1e7ebaaf24fe78142762fa222"
PINNED_REPRESENTATIVE_SELECTION_SEAL_SHA256 = "935245158fd87f7da2e8aac77060bb52a702e35732472c161910e0a888d7d0a7"
PINNED_SELECTION_IDENTITY_SHA256 = "4a3e5575b6c765da4c428a8db1b4a572d8adda2cd24be54963cc4ce8fa05f75d"

# These are execution machinery, not benchmark semantics.  A changed blob is
# allowed only in this closed set/prefix and is recorded as a deliberate repin.
# Workloads, definitions, solutions, and budget/config inputs must remain byte
# identical across the authority and execution trees.
EXECUTION_TOOL_PATHS = (
    "tools/local_labels/build_sol_representative_capture_matrix.py",
    "tools/local_labels/build_sol_representative_execution_matrix.py",
    "tools/local_labels/build_sol_workload_capture_matrix.py",
    "tools/local_labels/campaign_done_manifest.py",
    "tools/local_labels/campaign_preflight.py",
    "tools/local_labels/capture_toolchain_contract.py",
    "tools/local_labels/label_driver.sh",
    "tools/local_labels/metric_registry.json",
    "tools/local_labels/protected_paths.py",
    "tools/local_labels/sm86_capture_chunk.py",
    "tools/local_labels/sol_capture_environment.py",
    "tools/local_labels/sol_definition_equivalence.py",
    "tools/local_labels/sol_live_workload_coverage_inventory.py",
    "tools/local_labels/sol_reconstitution.py",
    "tools/local_labels/sol_shrink_apply.py",
    "tools/local_labels/sol_workload_coverage_inventory.py",
    "tools/local_labels/sol_workload_capture_campaign.py",
    "tools/local_labels/sol_workload_capture_chunk.py",
    "tools/cloud/trace_budgets.py",
    "workloads/simbench_bw/schemas/campaign_preflight_manifest.schema.json",
    "workloads/simbench_bw/schemas/campaign_preflight_manifest.v3.schema.json",
    "workloads/simbench_bw/schemas/sol_live_workload_coverage_inventory.schema.json",
    "workloads/simbench_bw/schemas/sol_representative_capture_matrix.schema.json",
    "workloads/simbench_bw/schemas/sol_representative_execution_matrix.schema.json",
    "workloads/simbench_bw/schemas/sol_workload_capture_chunk_done.schema.json",
    "workloads/simbench_bw/schemas/sol_workload_capture_matrix.schema.json",
)
EXECUTION_TOOL_PREFIXES = (
    "tools/sol_execbench/src/sol_execbench/",
)

REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT = {
    "inventory_policy": "requires_inventory_before_capture",
    "launch_scope": "all_in_profiler_roi",
    "trace_requirement": "every_inventory_launch_complete",
    "ncu_label_requirement": "every_inventory_launch_complete_cycles_only_binding",
    "control_requirement": "complete_or_origin_unresolved_typed_unavailable",
    "admission_policy": "trace_and_ncu_complete_control_typed_nonblocking",
}


class RepresentativeExecutionError(ValueError):
    """The representative execution authority is incomplete or inconsistent."""


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise RepresentativeExecutionError(detail)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Match the source matrix's compact contract hashing rule."""
    return source_builder.canonical_sha256(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, *, role: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepresentativeExecutionError(f"cannot read {role} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{role} root is not an object: {path}")
    return value, payload


def _read_regular_sibling(
    matrix_path: Path, *, name: Any, role: str,
) -> tuple[dict[str, Any], bytes]:
    """Read one exact, non-symlink sibling retained with an execution matrix."""
    _require(
        isinstance(name, str) and name and Path(name).name == name,
        f"{role} sibling name is unsafe: {name!r}",
    )
    try:
        matrix_stat = matrix_path.lstat()
        _require(
            stat.S_ISREG(matrix_stat.st_mode) and not matrix_path.is_symlink(),
            f"execution matrix is not a regular non-symlink file: {matrix_path}",
        )
        parent = matrix_path.resolve(strict=True).parent
        candidate = parent / name
        candidate_stat = candidate.lstat()
        _require(
            stat.S_ISREG(candidate_stat.st_mode) and not candidate.is_symlink(),
            f"{role} is not a regular non-symlink sibling: {candidate}",
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(candidate, flags)
        try:
            opened = os.fstat(descriptor)
            _require(
                stat.S_ISREG(opened.st_mode)
                and (opened.st_dev, opened.st_ino)
                == (candidate_stat.st_dev, candidate_stat.st_ino),
                f"{role} changed during open: {candidate}",
            )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read()
        finally:
            os.close(descriptor)
        value = json.loads(payload)
    except RepresentativeExecutionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepresentativeExecutionError(
            f"cannot read retained {role} sibling {name!r}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"retained {role} root is not an object")
    return value, payload


def load_runtime_upstream_operands(
    document: dict[str, Any], *, matrix_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    """Load the two immutable upstream payloads from the matrix directory."""
    source_pin = document.get("source_matrix") or {}
    selection_pin = document.get("representative_selection") or {}
    source, source_payload = _read_regular_sibling(
        matrix_path, name=source_pin.get("name"), role="source matrix",
    )
    representative, representative_payload = _read_regular_sibling(
        matrix_path,
        name=selection_pin.get("name"),
        role="representative selection",
    )
    return source, source_payload, representative, representative_payload


def _git(
    repo_root: Path, *args: str, text: bool = False,
) -> bytes | str:
    environment = os.environ.copy()
    for name in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_DIR",
        "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_PREFIX",
        "GIT_WORK_TREE",
    ):
        environment.pop(name, None)
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True,
        text=text, check=False, env=environment,
    )
    if result.returncode:
        stderr = result.stderr.strip() if text else result.stderr.decode(
            "utf-8", errors="replace"
        ).strip()
        raise RepresentativeExecutionError(
            f"git {' '.join(args)} failed ({result.returncode}): {stderr}"
        )
    return result.stdout


def git_head_state(repo_root: Path) -> tuple[str, list[str]]:
    head = str(_git(repo_root, "rev-parse", "HEAD", text=True)).strip()
    dirty = str(_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all",
        text=True,
    )).splitlines()
    _require(SHA40_RE.fullmatch(head) is not None, f"invalid Git HEAD: {head!r}")
    return head, dirty


def _git_blob(repo_root: Path, commit: str, path: str) -> tuple[str, str]:
    _require(
        path and not Path(path).is_absolute() and ".." not in Path(path).parts,
        f"unsafe Git dependency path: {path!r}",
    )
    blob = bytes(_git(repo_root, "show", f"{commit}:{path}"))
    blob_id = str(_git(
        repo_root, "rev-parse", f"{commit}:{path}", text=True,
    )).strip()
    _require(SHA40_RE.fullmatch(blob_id) is not None, f"invalid blob id: {path}")
    return blob_id, sha256_bytes(blob)


def _tree_paths(repo_root: Path, commit: str, prefix: str) -> set[str]:
    output = str(_git(
        repo_root, "ls-tree", "-r", "--name-only", commit, "--", prefix,
        text=True,
    ))
    return {line for line in output.splitlines() if line}


def _static_input_pins(variants: Iterable[dict[str, Any]], matrix: dict[str, Any]) -> dict[str, str]:
    """Return the closed semantic/config file set declared by selected variants."""
    pins: dict[str, str] = {}

    def add(path: Any, digest: Any, *, role: str) -> None:
        if not isinstance(path, str) or not path or not isinstance(digest, str):
            raise RepresentativeExecutionError(f"{role} pin is absent")
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise RepresentativeExecutionError(f"unsafe {role} path: {path!r}")
        if SHA256_RE.fullmatch(digest) is None:
            raise RepresentativeExecutionError(f"invalid {role} SHA-256: {path}")
        prior = pins.get(path)
        _require(prior in {None, digest}, f"conflicting {role} pins: {path}")
        pins[path] = digest

    def require_external_benchmark(path: Any, digest: Any, *, role: str) -> None:
        _require(
            isinstance(path, str)
            and path.startswith("tools/sol_execbench/data/benchmark/")
            and ".." not in Path(path).parts
            and isinstance(digest, str)
            and SHA256_RE.fullmatch(digest) is not None,
            f"invalid retained {role} pin: {path!r}",
        )

    for variant in variants:
        if variant.get("cohort") == "reduced_proxy":
            provenance = variant.get("reduced_provenance") or {}
            logical = (variant.get("invocation") or {}).get(
                "logical_workload"
            ) or {}
            external_workload = logical.get("external_workload_source")
            _require(
                isinstance(external_workload, dict)
                and external_workload.get("name")
                == variant.get("source_workload_file")
                and external_workload.get("sha256")
                == variant.get("source_workload_file_sha256"),
                "reduced workload is not bound to its retained external source",
            )
            add(
                variant.get("solution_file"),
                variant.get("solution_file_sha256"), role="solution",
            )
            require_external_benchmark(
                provenance.get("native_workload_file"),
                provenance.get("native_workload_file_sha256"),
                role="native workload behind retained reduced proxy",
            )
            require_external_benchmark(
                provenance.get("native_definition_file"),
                provenance.get("native_definition_file_sha256"),
                role="native definition behind retained reduced proxy",
            )
            external_definition = logical.get("external_definition_source")
            if isinstance(external_definition, dict):
                _require(
                    external_definition.get("name")
                    == variant.get("definition_file")
                    and external_definition.get("sha256")
                    == variant.get("definition_file_sha256"),
                    "reduced definition is not bound to its retained external source",
                )
            else:
                _require(
                    variant.get("definition_file")
                    == provenance.get("native_definition_file")
                    and variant.get("definition_file_sha256")
                    == provenance.get("native_definition_file_sha256"),
                    "reduced definition is neither native nor retained external evidence",
                )
        else:
            require_external_benchmark(
                variant.get("source_workload_file"),
                variant.get("source_workload_file_sha256"), role="workload",
            )
            require_external_benchmark(
                variant.get("definition_file"),
                variant.get("definition_file_sha256"), role="definition",
            )
            add(
                variant.get("solution_file"),
                variant.get("solution_file_sha256"), role="solution",
            )
        adapter = variant.get("adapter_contract") or {}
        flashinfer = adapter.get("flashinfer_corpus") or {}
        if isinstance(flashinfer, dict) and flashinfer.get("category") == "FIB":
            add(
                flashinfer.get("data_annex_manifest_path"),
                flashinfer.get("data_annex_manifest_sha256"),
                role="FlashInfer data-annex manifest",
            )
        budget = variant.get("trace_budget")
        _require(isinstance(budget, dict), "source trace budget is absent")
        add(budget.get("policy_path"), budget.get("policy_sha256"), role="budget policy")
        add(budget.get("trace_budgets_path"), budget.get("trace_budgets_sha256"), role="budget config")
    fingerprint = (matrix.get("input_pins") or {}).get("sol_native_env_fingerprint")
    _require(isinstance(fingerprint, dict), "native environment input pin is absent")
    add(fingerprint.get("path"), fingerprint.get("sha256"), role="native environment")
    return pins


def _external_runtime_pins(variants: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record non-Git executable/reduced inputs without pretending Git owns them."""
    records: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(kind: str, name: Any, digest: Any) -> None:
        if not isinstance(name, str) or not name or not isinstance(digest, str):
            return
        _require(SHA256_RE.fullmatch(digest) is not None, f"invalid external pin: {name}")
        key = kind, name, digest
        records[key] = {
            "kind": kind,
            "name": name,
            "sha256": digest,
            "verification": "exact_runtime_or_retained_owner_hash",
        }

    for variant in variants:
        invocation = variant.get("invocation") or {}
        add("runtime_executable", invocation.get("executable_path"), invocation.get("executable_sha256"))
        logical = invocation.get("logical_workload") or {}
        if variant.get("cohort") == "reduced_proxy":
            provenance = variant.get("reduced_provenance") or {}
            add(
                "native_workload_behind_reduced_proxy",
                provenance.get("native_workload_file"),
                provenance.get("native_workload_file_sha256"),
            )
            add(
                "native_definition_behind_reduced_proxy",
                provenance.get("native_definition_file"),
                provenance.get("native_definition_file_sha256"),
            )
        else:
            add(
                "sol_benchmark_workload",
                variant.get("source_workload_file"),
                variant.get("source_workload_file_sha256"),
            )
            add(
                "sol_benchmark_definition",
                variant.get("definition_file"),
                variant.get("definition_file_sha256"),
            )
        add(
            "sol_benchmark_tree",
            logical.get("benchmark_dir"),
            logical.get("benchmark_tree_sha256"),
        )
        for kind, field in (
            ("retained_reduced_workload", "external_workload_source"),
            ("retained_reduced_definition", "external_definition_source"),
        ):
            source = logical.get(field)
            if isinstance(source, dict):
                add(kind, source.get("name"), source.get("sha256"))
    return [records[key] for key in sorted(records)]


def build_carry_forward_proof(
    *, repo_root: Path, authority_commit: str, execution_commit: str,
    variants: list[dict[str, Any]], source_matrix: dict[str, Any],
) -> dict[str, Any]:
    """Prove semantic inputs unchanged and bind every changed execution tool."""
    _require(SHA40_RE.fullmatch(authority_commit) is not None, "authority commit invalid")
    _require(SHA40_RE.fullmatch(execution_commit) is not None, "execution commit invalid")
    semantic_pins = _static_input_pins(variants, source_matrix)
    semantic_rows: list[dict[str, Any]] = []
    for path, declared_sha in sorted(semantic_pins.items()):
        authority_blob, authority_sha = _git_blob(repo_root, authority_commit, path)
        execution_blob, execution_sha = _git_blob(repo_root, execution_commit, path)
        _require(
            declared_sha == authority_sha == execution_sha,
            f"semantic input changed or source pin is false: {path}; "
            f"declared={declared_sha}; authority={authority_sha}; execution={execution_sha}",
        )
        semantic_rows.append({
            "path": path,
            "role": "selected_semantic_or_config_input",
            "disposition": "byte_identical_carry_forward",
            "declared_sha256": declared_sha,
            "authority_blob": authority_blob,
            "execution_blob": execution_blob,
        })

    tools = set(EXECUTION_TOOL_PATHS)
    for prefix in EXECUTION_TOOL_PREFIXES:
        tools.update(_tree_paths(repo_root, authority_commit, prefix))
        tools.update(_tree_paths(repo_root, execution_commit, prefix))
    tool_rows: list[dict[str, Any]] = []
    for path in sorted(tools):
        try:
            authority_blob, authority_sha = _git_blob(repo_root, authority_commit, path)
        except RepresentativeExecutionError:
            authority_blob = authority_sha = None
        try:
            execution_blob, execution_sha = _git_blob(repo_root, execution_commit, path)
        except RepresentativeExecutionError:
            execution_blob = execution_sha = None
        _require(execution_sha is not None, f"required execution tool is absent: {path}")
        disposition = (
            "byte_identical_carry_forward"
            if authority_sha == execution_sha
            else (
                "deliberately_added_and_repinned"
                if authority_sha is None
                else "deliberately_updated_and_repinned"
            )
        )
        tool_rows.append({
            "path": path,
            "role": "execution_tool",
            "disposition": disposition,
            "authority_blob": authority_blob,
            "authority_sha256": authority_sha,
            "execution_blob": execution_blob,
            "execution_sha256": execution_sha,
        })

    proof = {
        "authority_commit": authority_commit,
        "execution_commit": execution_commit,
        "semantic_inputs": semantic_rows,
        "execution_tools": tool_rows,
        "external_runtime_inputs": _external_runtime_pins(variants),
    }
    proof["proof_sha256"] = canonical_sha256(proof)
    return proof


def _representative_budget(source: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(source, dict), "source trace budget is not an object")
    source_wall = source.get("max_wall_time_s")
    _require(type(source_wall) is int and source_wall > 0, "source wall-time budget invalid")
    result = copy.deepcopy(source)
    source_trace_bytes = source.get("max_trace_bytes")
    _require(
        type(source_trace_bytes) is int and source_trace_bytes > 0,
        "source trace-byte budget invalid",
    )
    result["source_max_trace_bytes"] = source_trace_bytes
    result["max_trace_bytes"] = min(
        source_trace_bytes, REPRESENTATIVE_MAX_TRACE_BYTES
    )
    result["source_max_wall_time_s"] = source_wall
    result["max_wall_time_s"] = min(source_wall, REPRESENTATIVE_MAX_WALL_TIME_S)
    result["derivation"] = "representative_min_source_and_900s_ceiling"
    result["timeout_disposition"] = "typed_measured_infeasibility_for_trace_proxy_planning"
    result["trace_limit_disposition"] = "typed_measured_infeasibility_for_trace_proxy_planning"
    _require(result.get("sampling_policy") == "off", "representative sampling must remain off")
    _require(result.get("partial_trace_allowed") is False, "partial representative traces forbidden")
    return result


def _variant_cell_contract(variant: dict[str, Any]) -> dict[str, Any]:
    adapter = variant.get("adapter_contract")
    contract: dict[str, Any] = {
        "identity": {"roster_source": "sol_execbench", "slug": variant.get("slug")},
        "category": variant.get("category"),
        "flashinfer_corpus": copy.deepcopy(
            adapter.get("flashinfer_corpus") if isinstance(adapter, dict) else None
        ),
        "launch_scope": variant.get("launch_scope"),
        "launch_completeness_contract": copy.deepcopy(
            variant.get("launch_completeness_contract")
        ),
        "invocation": copy.deepcopy(variant.get("invocation")),
        "trace_budget": copy.deepcopy(variant.get("trace_budget")),
        "control_bits": copy.deepcopy(variant.get("control_bits")),
    }
    cohort = variant.get("cohort")
    if cohort == "partial_native_row":
        contract["selected_diagnostic_sha256"] = variant.get("selected_diagnostic_sha256")
    elif cohort == "reduced_proxy":
        provenance = variant.get("reduced_provenance")
        contract["row_plan_sha256"] = (
            provenance.get("row_plan_sha256") if isinstance(provenance, dict) else None
        )
    else:
        _require(cohort == "native", f"unknown representative cohort: {cohort!r}")
    return contract


def rebuild_execution_variant(
    source: dict[str, Any], *, authority_commit: str, execution_commit: str,
    carry_forward_sha256: str,
) -> dict[str, Any]:
    authority_seal = source.get("variant_contract_sha256")
    authority_cell = source.get("supplemental_cell_contract_sha256")
    unsigned_source = {key: value for key, value in source.items() if key != "variant_contract_sha256"}
    _require(
        SHA256_RE.fullmatch(str(authority_seal or "")) is not None
        and canonical_sha256(unsigned_source) == authority_seal,
        f"source variant seal mismatch: {source.get('slug')}:{source.get('variant_id')}",
    )
    result = copy.deepcopy(source)
    source_launch = copy.deepcopy(result["launch_completeness_contract"])
    source_budget = copy.deepcopy(result["trace_budget"])
    result["authority_variant_contract_sha256"] = authority_seal
    result["authority_supplemental_cell_contract_sha256"] = authority_cell
    result["source_launch_completeness_contract"] = source_launch
    result["source_trace_budget"] = source_budget
    result["launch_completeness_contract"] = copy.deepcopy(
        REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    )
    result["trace_budget"] = _representative_budget(source_budget)
    result["execution_receipt_sha256"] = canonical_sha256({
        "authority_commit": authority_commit,
        "authority_variant_contract_sha256": authority_seal,
        "carry_forward_proof_sha256": carry_forward_sha256,
        "execution_commit": execution_commit,
        "execution_profile": EXECUTION_PROFILE,
        "representative_launch_completeness_contract": (
            REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
        ),
        "representative_max_wall_time_s": REPRESENTATIVE_MAX_WALL_TIME_S,
    })
    result["supplemental_cell_contract_sha256"] = canonical_sha256(
        _variant_cell_contract(result)
    )
    result.pop("variant_contract_sha256", None)
    result["variant_contract_sha256"] = canonical_sha256(result)
    return result


def rebuild_execution_row(
    source: dict[str, Any], *, variant: dict[str, Any],
    authority_row_sha256: str, execution_commit: str,
) -> dict[str, Any]:
    result = copy.deepcopy(source)
    _require(
        canonical_sha256(source) == authority_row_sha256,
        f"source row hash mismatch: {source.get('key')!r}",
    )
    result["authority_row_contract_sha256"] = authority_row_sha256
    result["authority_variant_contract_sha256"] = source[
        "variant_contract_sha256"
    ]
    result["variant_contract_sha256"] = variant["variant_contract_sha256"]
    result["launch_completeness_contract"] = copy.deepcopy(
        REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    )
    result["execution_row_contract_sha256"] = canonical_sha256({
        "authority_row_contract_sha256": authority_row_sha256,
        "execution_commit": execution_commit,
        "execution_profile": EXECUTION_PROFILE,
        "execution_variant_contract_sha256": variant["variant_contract_sha256"],
        "row_key": result["key"],
    })
    return result


def _recover_authority_variant(execution_variant: dict[str, Any]) -> dict[str, Any]:
    """Reverse only the declared execution projection back to its authority row."""
    result = copy.deepcopy(execution_variant)
    authority_variant_sha256 = result.pop("authority_variant_contract_sha256", None)
    authority_cell_sha256 = result.pop(
        "authority_supplemental_cell_contract_sha256", None
    )
    source_launch = result.pop("source_launch_completeness_contract", None)
    source_budget = result.pop("source_trace_budget", None)
    result.pop("execution_receipt_sha256", None)
    result["launch_completeness_contract"] = source_launch
    result["trace_budget"] = source_budget
    result["supplemental_cell_contract_sha256"] = authority_cell_sha256
    result.pop("variant_contract_sha256", None)
    _require(
        SHA256_RE.fullmatch(str(authority_variant_sha256 or "")) is not None
        and canonical_sha256(result) == authority_variant_sha256,
        "execution variant does not reverse to its authority seal: "
        f"{execution_variant.get('slug')}:{execution_variant.get('variant_id')}",
    )
    result["variant_contract_sha256"] = authority_variant_sha256
    return result


def _recover_authority_row(
    execution_row: dict[str, Any], *, execution_variant: dict[str, Any]
) -> dict[str, Any]:
    """Reverse only the declared execution-row projection to its authority row."""
    result = copy.deepcopy(execution_row)
    authority_row_sha256 = result.pop("authority_row_contract_sha256", None)
    authority_variant_sha256 = result.pop(
        "authority_variant_contract_sha256", None
    )
    result.pop("execution_row_contract_sha256", None)
    result["variant_contract_sha256"] = authority_variant_sha256
    result["launch_completeness_contract"] = copy.deepcopy(
        execution_variant.get("source_launch_completeness_contract")
    )
    _require(
        SHA256_RE.fullmatch(str(authority_row_sha256 or "")) is not None
        and canonical_sha256(result) == authority_row_sha256,
        "execution row does not reverse to its authority seal: "
        f"{execution_row.get('key')!r}",
    )
    return result


def _interleaved_chunks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce cohort-pure chunks in cross-cohort waves for useful pilots."""
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort"]].append(copy.deepcopy(row["key"]))
    groups: dict[str, list[list[dict[str, Any]]]] = {
        cohort: [members[start:start + ROWS_PER_CHUNK]
                 for start in range(0, len(members), ROWS_PER_CHUNK)]
        for cohort, members in by_cohort.items()
    }
    result: list[dict[str, Any]] = []
    cohort_order = ("native", "reduced_proxy", "partial_native_row")
    for wave in range(max((len(value) for value in groups.values()), default=0)):
        for cohort in cohort_order:
            cohorts = groups.get(cohort, [])
            if wave >= len(cohorts):
                continue
            members = cohorts[wave]
            label = cohort.replace("_", "-")
            result.append({
                "chunk_id": f"sol-representative-{label}-{wave:03d}",
                "cohort": cohort,
                "tier": "representative",
                "ordinal": len(result) + 1,
                "row_count": len(members),
                "membership_sha256": canonical_sha256(members),
                "rows": members,
            })
    return result


def _execution_counts(
    *, representative: dict[str, Any], variants: list[dict[str, Any]],
    rows: list[dict[str, Any]], chunks: list[dict[str, Any]],
) -> dict[str, int]:
    cohort_rows = Counter(row["cohort"] for row in rows)
    cohort_variants = Counter(row["cohort"] for row in variants)
    selection_counts = representative.get("counts") or {}
    return {
        "problems": len({row["key"]["slug"] for row in rows}),
        "variants": len(variants),
        "rows": len(rows),
        "chunks": len(chunks),
        "native_variants": cohort_variants["native"],
        "partial_native_variants": cohort_variants["partial_native_row"],
        "reduced_variants": cohort_variants["reduced_proxy"],
        "native_rows": cohort_rows["native"],
        "partial_native_rows": cohort_rows["partial_native_row"],
        "reduced_proxy_rows": cohort_rows["reduced_proxy"],
        "application_coverage_anchors": selection_counts[
            "application_coverage_anchors"
        ],
        "native_comparison_companions": selection_counts[
            "native_comparison_companions"
        ],
        "architecture_inapplicable_problems": selection_counts[
            "architecture_inapplicable_problems"
        ],
        "architecture_inapplicable_rows": selection_counts[
            "architecture_inapplicable_rows"
        ],
    }


def _selection_identity(selection: dict[str, Any]) -> tuple[str, str, str]:
    key = selection.get("source_matrix_row_key")
    _require(isinstance(key, dict), "selection row key is absent")
    identity = key.get("slug"), key.get("source_workload_uuid"), key.get("variant_id")
    _require(all(isinstance(value, str) and value for value in identity), "selection key invalid")
    return identity  # type: ignore[return-value]


def _selected_source_pairs(
    source_matrix: dict[str, Any], representative: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Resolve the exact selected rows and variants from immutable authorities."""
    source_variants: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in source_matrix.get("variants") or []:
        key = variant.get("slug"), variant.get("variant_id")
        _require(key not in source_variants, f"duplicate source variant: {key!r}")
        source_variants[key] = variant
    source_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in source_matrix.get("rows") or []:
        key = row.get("key") or {}
        identity = (
            key.get("slug"), key.get("source_workload_uuid"),
            key.get("variant_id"),
        )
        _require(identity not in source_rows, f"duplicate source row: {identity!r}")
        source_rows[identity] = row

    selected = representative.get("selections")
    _require(
        isinstance(selected, list) and selected,
        "representative selections absent",
    )
    result: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    identities: set[tuple[str, str, str]] = set()
    for selection in selected:
        identity = _selection_identity(selection)
        _require(
            identity not in identities,
            f"duplicate representative selection: {identity!r}",
        )
        identities.add(identity)
        row = source_rows.get(identity)
        variant = source_variants.get((identity[0], identity[2]))
        _require(
            row is not None and variant is not None,
            f"selection absent from source: {identity!r}",
        )
        _require(
            canonical_sha256(row) == selection.get("source_matrix_row_sha256")
            and variant.get("variant_contract_sha256")
            == selection.get("source_matrix_variant_sha256"),
            f"selection source pin mismatch: {identity!r}",
        )
        result.append((selection, row, variant))
    return result


def _validate_pinned_upstream_authorities(
    *, source_matrix: dict[str, Any], source_matrix_payload: bytes,
    representative: dict[str, Any], representative_payload: bytes,
) -> str:
    """Verify that parsed upstream objects are the two exact pinned payloads."""
    try:
        parsed_source = json.loads(source_matrix_payload)
        parsed_representative = json.loads(representative_payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RepresentativeExecutionError(
            f"upstream authority payload is not JSON: {exc}"
        ) from exc
    _require(
        parsed_source == source_matrix and parsed_representative == representative,
        "upstream parsed objects do not match their retained bytes",
    )
    authority_commit = source_matrix.get("source_commit")
    _require(
        SHA40_RE.fullmatch(str(authority_commit or "")) is not None,
        "source commit invalid",
    )
    _require(
        authority_commit == PINNED_AUTHORITY_COMMIT
        and sha256_bytes(source_matrix_payload) == PINNED_SOURCE_MATRIX_SHA256
        and source_matrix.get("matrix_seal_sha256")
        == PINNED_SOURCE_MATRIX_SEAL_SHA256,
        "source matrix is not the pinned e387 campaign authority",
    )
    _require(
        representative.get("artifact_id")
        == "sm86_sol_representative_capture_matrix_v1"
        and representative.get("source_commit") == authority_commit,
        "representative selection identity/source commit mismatch",
    )
    _require(
        sha256_bytes(representative_payload)
        == PINNED_REPRESENTATIVE_SELECTION_SHA256
        and representative.get("matrix_seal_sha256")
        == PINNED_REPRESENTATIVE_SELECTION_SEAL_SHA256
        and (representative.get("selection_receipt") or {}).get(
            "selection_identity_sha256"
        ) == PINNED_SELECTION_IDENTITY_SHA256,
        "representative selection is not the pinned 216-row authority",
    )
    source_seal = source_matrix.get("matrix_seal_sha256")
    _require(
        source_builder.canonical_sha256({
            key: value for key, value in source_matrix.items()
            if key != "matrix_seal_sha256"
        }) == source_seal,
        "source matrix seal mismatch",
    )
    rep_seal = representative.get("matrix_seal_sha256")
    _require(
        selector.canonical_sha256({
            key: value for key, value in representative.items()
            if key != "matrix_seal_sha256"
        }) == rep_seal,
        "representative selection seal mismatch",
    )
    return authority_commit


def build_execution_matrix(
    *, source_matrix: dict[str, Any], source_matrix_payload: bytes,
    representative: dict[str, Any], representative_payload: bytes,
    repo_root: Path = REPO_ROOT, execution_commit: str | None = None,
    require_clean_head: bool = True, enforce_current_counts: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    authority_commit = _validate_pinned_upstream_authorities(
        source_matrix=source_matrix,
        source_matrix_payload=source_matrix_payload,
        representative=representative,
        representative_payload=representative_payload,
    )
    source_seal = source_matrix.get("matrix_seal_sha256")
    rep_seal = representative.get("matrix_seal_sha256")

    head, dirty = git_head_state(root)
    execution_commit = execution_commit or head
    _require(head == execution_commit, f"execution commit is not HEAD: {execution_commit} != {head}")
    if require_clean_head:
        _require(not dirty, f"execution HEAD is dirty: {dirty!r}")

    selected_pairs = _selected_source_pairs(source_matrix, representative)

    selected_source_variants = [pair[2] for pair in selected_pairs]
    carry_forward = build_carry_forward_proof(
        repo_root=root, authority_commit=authority_commit,
        execution_commit=execution_commit, variants=selected_source_variants,
        source_matrix=source_matrix,
    )
    execution_variants: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    for selection, source_row, source_variant in selected_pairs:
        variant = rebuild_execution_variant(
            source_variant, authority_commit=authority_commit,
            execution_commit=execution_commit,
            carry_forward_sha256=carry_forward["proof_sha256"],
        )
        row = rebuild_execution_row(
            source_row, variant=variant,
            authority_row_sha256=selection["source_matrix_row_sha256"],
            execution_commit=execution_commit,
        )
        execution_variants.append(variant)
        execution_rows.append(row)

    chunks = _interleaved_chunks(execution_rows)
    declared_retention_bytes = sum(
        variant["trace_budget"]["max_trace_bytes"]
        for variant in execution_variants
    )
    metadata_margin_gib = 8
    gib = 1024 ** 3
    required_retention_gib = (
        (declared_retention_bytes + gib - 1) // gib + metadata_margin_gib
    )
    counts = _execution_counts(
        representative=representative,
        variants=execution_variants,
        rows=execution_rows,
        chunks=chunks,
    )
    if enforce_current_counts:
        _require(counts == {
            "problems": 201, "variants": 216, "rows": 216, "chunks": 15,
            "native_variants": 178, "partial_native_variants": 15,
            "reduced_variants": 23, "native_rows": 178,
            "partial_native_rows": 15, "reduced_proxy_rows": 23,
            "application_coverage_anchors": 201,
            "native_comparison_companions": 15,
            "architecture_inapplicable_problems": 34,
            "architecture_inapplicable_rows": 537,
        }, f"representative execution counts changed: {counts!r}")

    document: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "execution_profile": EXECUTION_PROFILE,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "source_commit": execution_commit,
        "authority_source_commit": authority_commit,
        "source_matrix": {
            "name": Path("sol_workload_capture_matrix.live.v2.json").name,
            "sha256": sha256_bytes(source_matrix_payload),
            "matrix_seal_sha256": source_seal,
        },
        "representative_selection": {
            "name": Path("sol_representative_capture_matrix.v1.json").name,
            "sha256": sha256_bytes(representative_payload),
            "matrix_seal_sha256": rep_seal,
            "selection_identity_sha256": representative[
                "selection_receipt"
            ]["selection_identity_sha256"],
        },
        "authority_input_pins": copy.deepcopy(representative["input_pins"]),
        "carry_forward_proof": carry_forward,
        "representative_capture_contract": copy.deepcopy(
            selector.REPRESENTATIVE_CAPTURE_CONTRACT
        ),
        "launch_completeness_contract": copy.deepcopy(
            REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
        ),
        "timeout_contract": {
            "ceiling_seconds": REPRESENTATIVE_MAX_WALL_TIME_S,
            "derivation": "min(source_max_wall_time_s, ceiling_seconds)",
            "failure_disposition": "typed_measured_infeasibility_for_trace_proxy_planning",
            "sampling_policy": "off",
            "partial_trace_allowed": False,
        },
        "aggregate_retention_contract": {
            "selected_max_trace_bytes_sum": declared_retention_bytes,
            "metadata_margin_gib": metadata_margin_gib,
            "required_retention_budget_gib": required_retention_gib,
            "runtime_requirement": (
                "external_aggregate_budget_at_least_required_retention"
            ),
            "raw_staging_and_free_space_checks_unchanged": True,
        },
        "input_pins": copy.deepcopy(source_matrix["input_pins"]),
        "canonical_matrix": copy.deepcopy(source_matrix["canonical_matrix"]),
        "canonical_authority": copy.deepcopy(source_matrix["canonical_authority"]),
        "runnability_authority": copy.deepcopy(source_matrix["runnability_authority"]),
        "supplemental_budget_policy": copy.deepcopy(source_matrix["supplemental_budget_policy"]),
        "reduced_proxy_authority": copy.deepcopy(source_matrix["reduced_proxy_authority"]),
        "claim_policy": copy.deepcopy(source_matrix["claim_policy"]),
        "chunk_policy": {
            "rows_per_chunk": ROWS_PER_CHUNK,
            "cohort_pure": True,
            "order": "interleaved_cohort_waves_native_reduced_partial",
            "pilot_contract": "first_three_chunks_cover_native_reduced_partial",
        },
        "counts": counts,
        "architecture_exclusions": copy.deepcopy(
            representative["architecture_exclusions"]
        ),
        "variants": execution_variants,
        "rows": execution_rows,
        "chunks": chunks,
    }
    document["matrix_seal_sha256"] = canonical_sha256(document)
    validate_execution_matrix(
        document, repo_root=root, require_clean_head=require_clean_head,
        enforce_current_counts=enforce_current_counts,
        source_matrix=source_matrix,
        source_matrix_payload=source_matrix_payload,
        representative=representative,
        representative_payload=representative_payload,
    )
    return document


def validate_execution_matrix(
    document: dict[str, Any], *, repo_root: Path = REPO_ROOT,
    require_clean_head: bool = True, enforce_current_counts: bool = True,
    source_matrix: dict[str, Any] | None = None,
    source_matrix_payload: bytes | None = None,
    representative: dict[str, Any] | None = None,
    representative_payload: bytes | None = None,
    matrix_path: Path | None = None,
) -> None:
    """Validate against the exact retained full matrix and selection bytes."""
    root = Path(repo_root).resolve()
    supplied = (
        source_matrix, source_matrix_payload,
        representative, representative_payload,
    )
    if matrix_path is not None:
        _require(
            all(value is None for value in supplied),
            "provide either matrix_path or explicit upstream operands, not both",
        )
        (
            source_matrix, source_matrix_payload,
            representative, representative_payload,
        ) = load_runtime_upstream_operands(document, matrix_path=matrix_path)
    else:
        _require(
            all(value is not None for value in supplied),
            "exact source-matrix and representative-selection operands are required",
        )
    assert source_matrix is not None
    assert source_matrix_payload is not None
    assert representative is not None
    assert representative_payload is not None
    authority_commit = _validate_pinned_upstream_authorities(
        source_matrix=source_matrix,
        source_matrix_payload=source_matrix_payload,
        representative=representative,
        representative_payload=representative_payload,
    )
    try:
        schema = json.loads((root / SCHEMA_PATH).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(document, schema)
    except (OSError, UnicodeError, json.JSONDecodeError, jsonschema.ValidationError,
            jsonschema.SchemaError) as exc:
        raise RepresentativeExecutionError(f"execution matrix schema invalid: {exc}") from exc
    seal = document.get("matrix_seal_sha256")
    _require(
        canonical_sha256({key: value for key, value in document.items()
                          if key != "matrix_seal_sha256"}) == seal,
        "execution matrix seal mismatch",
    )
    _require(
        document.get("authority_source_commit") == PINNED_AUTHORITY_COMMIT
        and document.get("source_matrix") == {
            "name": "sol_workload_capture_matrix.live.v2.json",
            "sha256": PINNED_SOURCE_MATRIX_SHA256,
            "matrix_seal_sha256": PINNED_SOURCE_MATRIX_SEAL_SHA256,
        }
        and document.get("representative_selection") == {
            "name": "sol_representative_capture_matrix.v1.json",
            "sha256": PINNED_REPRESENTATIVE_SELECTION_SHA256,
            "matrix_seal_sha256": PINNED_REPRESENTATIVE_SELECTION_SEAL_SHA256,
            "selection_identity_sha256": PINNED_SELECTION_IDENTITY_SHA256,
        },
        "execution matrix upstream authority pins changed",
    )
    head, dirty = git_head_state(root)
    _require(head == document.get("source_commit"), "execution matrix is not bound to HEAD")
    if require_clean_head:
        _require(not dirty, f"execution repository is dirty: {dirty!r}")
    variants = document.get("variants") or []
    proof = document.get("carry_forward_proof") or {}
    proof_seal = proof.get("proof_sha256")
    _require(
        canonical_sha256({key: value for key, value in proof.items()
                          if key != "proof_sha256"}) == proof_seal,
        "carry-forward proof seal mismatch",
    )
    _require(
        proof.get("authority_commit") == document.get("authority_source_commit")
        and proof.get("execution_commit") == document.get("source_commit"),
        "carry-forward commit binding mismatch",
    )
    semantic_rows = proof.get("semantic_inputs") or []
    expected_semantic = _static_input_pins(
        document.get("variants") or [], document
    )
    _require(
        [row.get("path") for row in semantic_rows]
        == sorted(expected_semantic),
        "carry-forward semantic path set is not dependency-closed",
    )
    for row in semantic_rows:
        path = row.get("path")
        _authority_blob, authority_sha = _git_blob(root, proof["authority_commit"], path)
        _execution_blob, execution_sha = _git_blob(root, proof["execution_commit"], path)
        _require(
            row.get("disposition") == "byte_identical_carry_forward"
            and row.get("declared_sha256") == authority_sha == execution_sha,
            f"carry-forward semantic input drift: {path}",
        )
    expected_tools = set(EXECUTION_TOOL_PATHS)
    for prefix in EXECUTION_TOOL_PREFIXES:
        expected_tools.update(_tree_paths(root, proof["authority_commit"], prefix))
        expected_tools.update(_tree_paths(root, proof["execution_commit"], prefix))
    tool_rows = proof.get("execution_tools") or []
    _require(
        [row.get("path") for row in tool_rows] == sorted(expected_tools),
        "carry-forward execution-tool path set is not dependency-closed",
    )
    for row in tool_rows:
        path = row.get("path")
        _blob, execution_sha = _git_blob(root, proof["execution_commit"], path)
        _require(row.get("execution_sha256") == execution_sha,
                 f"execution tool repin mismatch: {path}")
    _require(
        proof.get("external_runtime_inputs")
        == _external_runtime_pins(variants),
        "external runtime input closure mismatch",
    )
    selected_pairs = _selected_source_pairs(source_matrix, representative)
    selected_source_variants = [pair[2] for pair in selected_pairs]
    expected_proof = build_carry_forward_proof(
        repo_root=root,
        authority_commit=authority_commit,
        execution_commit=document["source_commit"],
        variants=selected_source_variants,
        source_matrix=source_matrix,
    )
    _require(
        proof == expected_proof,
        "carry-forward proof is not the exact recomputed dependency closure",
    )

    expected_variants: list[dict[str, Any]] = []
    expected_rows: list[dict[str, Any]] = []
    for selection, source_row, source_variant in selected_pairs:
        expected_variant = rebuild_execution_variant(
            source_variant,
            authority_commit=authority_commit,
            execution_commit=document["source_commit"],
            carry_forward_sha256=expected_proof["proof_sha256"],
        )
        expected_row = rebuild_execution_row(
            source_row,
            variant=expected_variant,
            authority_row_sha256=selection["source_matrix_row_sha256"],
            execution_commit=document["source_commit"],
        )
        expected_variants.append(expected_variant)
        expected_rows.append(expected_row)
    _require(
        variants == expected_variants,
        "execution variants are not the exact deterministic upstream projection",
    )
    _require(
        document.get("rows") == expected_rows,
        "execution rows are not the exact deterministic upstream projection",
    )
    expected_chunks = _interleaved_chunks(expected_rows)
    _require(
        document.get("chunks") == expected_chunks,
        "execution chunks are not the exact deterministic upstream projection",
    )
    _require(
        document.get("counts") == _execution_counts(
            representative=representative,
            variants=expected_variants,
            rows=expected_rows,
            chunks=expected_chunks,
        ),
        "execution counts are not the exact deterministic upstream projection",
    )
    for field in (
        "input_pins", "canonical_matrix", "canonical_authority",
        "runnability_authority", "supplemental_budget_policy",
        "reduced_proxy_authority", "claim_policy",
    ):
        _require(
            document.get(field) == source_matrix.get(field),
            f"execution top-level source projection changed: {field}",
        )
    _require(
        document.get("authority_input_pins") == representative.get("input_pins")
        and document.get("architecture_exclusions")
        == representative.get("architecture_exclusions"),
        "execution representative-selection projection changed",
    )

    rows = document.get("rows") or []
    chunks = document.get("chunks") or []
    counts = document.get("counts") or {}
    _require(counts.get("variants") == len(variants)
             and counts.get("rows") == len(rows)
             and counts.get("chunks") == len(chunks), "execution count mismatch")
    variants_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in variants:
        key = variant.get("slug"), variant.get("variant_id")
        _require(key not in variants_by_key, f"duplicate execution variant: {key!r}")
        unsigned = {name: value for name, value in variant.items()
                    if name != "variant_contract_sha256"}
        _require(canonical_sha256(unsigned) == variant.get("variant_contract_sha256"),
                 f"execution variant seal mismatch: {key!r}")
        _require(canonical_sha256(_variant_cell_contract(variant))
                 == variant.get("supplemental_cell_contract_sha256"),
                 f"execution cell contract mismatch: {key!r}")
        authority_variant = _recover_authority_variant(variant)
        _require(
            canonical_sha256(_variant_cell_contract(authority_variant))
            == variant.get("authority_supplemental_cell_contract_sha256"),
            f"authority cell contract mismatch: {key!r}",
        )
        expected_execution_receipt = canonical_sha256({
            "authority_commit": document["authority_source_commit"],
            "authority_variant_contract_sha256": variant[
                "authority_variant_contract_sha256"
            ],
            "carry_forward_proof_sha256": proof["proof_sha256"],
            "execution_commit": document["source_commit"],
            "execution_profile": EXECUTION_PROFILE,
            "representative_launch_completeness_contract": (
                REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
            ),
            "representative_max_wall_time_s": REPRESENTATIVE_MAX_WALL_TIME_S,
        })
        _require(
            variant.get("execution_receipt_sha256")
            == expected_execution_receipt,
            f"execution variant receipt mismatch: {key!r}",
        )
        budget = variant.get("trace_budget") or {}
        source_budget = variant.get("source_trace_budget") or {}
        _require(
            budget.get("max_wall_time_s")
            == min(source_budget.get("max_wall_time_s", 0), REPRESENTATIVE_MAX_WALL_TIME_S)
            and budget.get("max_trace_bytes")
            == min(source_budget.get("max_trace_bytes", 0), REPRESENTATIVE_MAX_TRACE_BYTES)
            and budget.get("source_max_trace_bytes")
            == source_budget.get("max_trace_bytes")
            and budget.get("sampling_policy") == "off"
            and budget.get("partial_trace_allowed") is False,
            f"representative budget invalid: {key!r}",
        )
        variants_by_key[key] = variant
    row_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        key_object = row.get("key") or {}
        key = key_object.get("slug"), key_object.get("source_workload_uuid"), key_object.get("variant_id")
        variant = variants_by_key.get((key[0], key[2]))
        _require(key not in row_keys and variant is not None, f"execution row closure invalid: {key!r}")
        _require(row.get("variant_contract_sha256") == variant.get("variant_contract_sha256"),
                 f"execution row variant mismatch: {key!r}")
        expected_row_receipt = canonical_sha256({
            "authority_row_contract_sha256": row.get("authority_row_contract_sha256"),
            "execution_commit": document["source_commit"],
            "execution_profile": EXECUTION_PROFILE,
            "execution_variant_contract_sha256": variant["variant_contract_sha256"],
            "row_key": row["key"],
        })
        _require(row.get("execution_row_contract_sha256") == expected_row_receipt,
                 f"execution row receipt mismatch: {key!r}")
        _recover_authority_row(row, execution_variant=variant)
        row_keys.add(key)
    scheduled: list[tuple[str, str, str]] = []
    for ordinal, chunk in enumerate(chunks, start=1):
        members = chunk.get("rows") or []
        _require(
            chunk.get("ordinal") == ordinal
            and chunk.get("row_count") == len(members)
            and 0 < len(members) <= ROWS_PER_CHUNK
            and chunk.get("membership_sha256") == canonical_sha256(members),
            f"execution chunk invalid: {chunk.get('chunk_id')}",
        )
        for member in members:
            key = member.get("slug"), member.get("source_workload_uuid"), member.get("variant_id")
            _require(key in row_keys, f"execution chunk member absent: {key!r}")
            scheduled.append(key)
    _require(len(scheduled) == len(set(scheduled)) and set(scheduled) == row_keys,
             "execution chunk partition is not exact")
    if len(chunks) >= 3:
        _require(
            [chunk["cohort"] for chunk in chunks[:3]]
            == ["native", "reduced_proxy", "partial_native_row"],
            "first representative pilot wave is not cross-cohort",
        )
    if enforce_current_counts:
        _require(
            counts.get("rows") == 216 and counts.get("chunks") == 15
            and counts.get("architecture_inapplicable_problems") == 34
            and counts.get("architecture_inapplicable_rows") == 537,
            "fixed representative counts changed",
        )
    retention = document.get("aggregate_retention_contract") or {}
    selected_max = sum(
        variant["trace_budget"]["max_trace_bytes"] for variant in variants
    )
    required_gib = (selected_max + (1024 ** 3) - 1) // (1024 ** 3) + 8
    _require(
        retention.get("selected_max_trace_bytes_sum") == selected_max
        and retention.get("required_retention_budget_gib") == required_gib
        and retention.get("metadata_margin_gib") == 8,
        "aggregate representative retention contract mismatch",
    )


def _load_and_build(args: argparse.Namespace) -> dict[str, Any]:
    source, source_payload = _read_object(args.source_matrix, role="source workload matrix")
    representative, representative_payload = _read_object(
        args.representative_selection, role="representative selection"
    )
    # Validate both upstream authorities before projecting their selected bytes.
    try:
        source_schema = json.loads((REPO_ROOT / source_builder.SCHEMA_PATH).read_text())
        selection_schema = json.loads((REPO_ROOT / selector.SCHEMA_PATH).read_text())
        jsonschema.validate(source, source_schema)
        jsonschema.validate(representative, selection_schema)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise RepresentativeExecutionError(f"upstream authority schema invalid: {exc}") from exc
    return build_execution_matrix(
        source_matrix=source, source_matrix_payload=source_payload,
        representative=representative, representative_payload=representative_payload,
        repo_root=REPO_ROOT, execution_commit=args.execution_commit,
        require_clean_head=True, enforce_current_counts=True,
    )


def _emit_fresh_external(path: Path, payload: bytes, *, repo_root: Path) -> None:
    """Create one fresh non-symlink artifact outside every Git worktree."""
    _require(path.is_absolute(), "execution matrix output must be absolute")
    parent = path.parent
    _require(parent.is_dir() and not parent.is_symlink(), "output parent is not a regular directory")
    resolved_parent = parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(repo_root.resolve())
    except ValueError:
        pass
    else:
        raise RepresentativeExecutionError("execution matrix output is repository-local")
    probe = subprocess.run(
        ["git", "-C", str(resolved_parent), "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True, check=False,
    )
    _require(
        not (probe.returncode == 0 and probe.stdout.strip() == "true"),
        "execution matrix output is inside a Git worktree",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-matrix", type=Path, required=True)
    parser.add_argument("--representative-selection", type=Path, required=True)
    parser.add_argument("--execution-commit")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", type=Path)
    mode.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    try:
        document = _load_and_build(args)
        payload = canonical_bytes(document)
        target = args.emit if args.emit is not None else args.check
        assert target is not None
        if args.check is not None:
            _require(target.read_bytes() == payload, "execution matrix regeneration mismatch")
            print(
                "SOL representative execution matrix verified: "
                f"rows={document['counts']['rows']} chunks={document['counts']['chunks']}"
            )
        else:
            _emit_fresh_external(target, payload, repo_root=REPO_ROOT)
            print(
                "SOL representative execution matrix emitted: "
                f"rows={document['counts']['rows']} chunks={document['counts']['chunks']} -> {target}"
            )
        return 0
    except (RepresentativeExecutionError, OSError) as exc:
        print(f"build_sol_representative_execution_matrix: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
