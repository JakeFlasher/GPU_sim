#!/usr/bin/env python3
"""Check that the SM86 handoff prose and summary match closure artifacts.

The live campaign has several intentionally different cp.async command
variants.  This small read-only validator keeps the human-readable handoff
from silently drifting away from the machine-readable closure report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_simulator_live_baseline_20260829_v2"
EXPECTED_IDS = (
    "cp_async_canary",
    "gemm_hmma_bf16_sm80",
    "gemm_hmma_fp16_sm80",
    "gemm_imma_int8_sm80",
    "gemm_sparse_2to4_bf16_sm80",
    "flash_attn_fwd_bf16_sm80",
)
TABLE_BEGIN = "<!-- SM86_HANDOFF_RESIDUAL_TABLE_BEGIN -->"
TABLE_END = "<!-- SM86_HANDOFF_RESIDUAL_TABLE_END -->"
TABLE_COLUMNS = (
    "cell",
    "ncu_gpc",
    "ncu_sm",
    "ncu_inst",
    "sim_completion",
    "sm_gap",
    "gpc_gap",
    "launch_gap",
    "dirty_writeback",
    "coherence",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}: boolean is not an integer")
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: malformed integer {value!r}") from exc


def _bool(value: object, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{label}: expected true/false")
    return normalized == "true"


def _parse_table(readme: str) -> dict[str, dict[str, object]]:
    try:
        body = readme.split(TABLE_BEGIN, 1)[1].split(TABLE_END, 1)[0]
    except IndexError as exc:
        raise ValueError("README residual table markers are missing") from exc
    lines = [line.strip() for line in body.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        raise ValueError("README residual table is missing rows")
    header = [part.strip().lower() for part in lines[0].strip("|").split("|")]
    if tuple(header) != TABLE_COLUMNS:
        raise ValueError(f"README residual table columns differ: {header!r}")
    rows: dict[str, dict[str, object]] = {}
    for line in lines[2:]:
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != len(TABLE_COLUMNS):
            raise ValueError("README residual table row has the wrong column count")
        cell = parts[0]
        if cell in rows:
            raise ValueError(f"README residual table duplicates {cell}")
        rows[cell] = {
            "cell": cell,
            "ncu_gpc": _int(parts[1], f"README/{cell}/ncu_gpc"),
            "ncu_sm": _int(parts[2], f"README/{cell}/ncu_sm"),
            "ncu_inst": _int(parts[3], f"README/{cell}/ncu_inst"),
            "sim_completion": _int(parts[4], f"README/{cell}/sim_completion"),
            "sm_gap": _int(parts[5], f"README/{cell}/sm_gap"),
            "gpc_gap": _int(parts[6], f"README/{cell}/gpc_gap"),
            "launch_gap": _int(parts[7], f"README/{cell}/launch_gap"),
            "dirty_writeback": _bool(parts[8], f"README/{cell}/dirty_writeback"),
            "coherence": _bool(parts[9], f"README/{cell}/coherence"),
        }
    return rows


def _closure_expected(closure: Mapping[str, Any]) -> dict[str, dict[str, object]]:
    rows = closure.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_IDS):
        raise ValueError("closure report must contain exactly six rows")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("cell") not in EXPECTED_IDS:
            raise ValueError("closure report contains an unknown or malformed cell")
        cell = str(row["cell"])
        live = row.get("live")
        simulator = row.get("simulator")
        residual = row.get("residual")
        if not isinstance(live, Mapping) or not isinstance(simulator, Mapping) or not isinstance(residual, Mapping):
            raise ValueError(f"closure/{cell}: live, simulator, and residual sections are required")
        baseline = live.get("baseline")
        if not isinstance(baseline, Mapping):
            raise ValueError(f"closure/{cell}: baseline metrics are missing")
        tails = simulator.get("tails")
        vector = residual.get("selection_vector")
        if not isinstance(tails, Mapping) or not isinstance(vector, Mapping):
            raise ValueError(f"closure/{cell}: typed tails or selection vector is missing")
        result[cell] = {
            "cell": cell,
            "ncu_gpc": _int(baseline.get("gpc__cycles_elapsed.max"), f"closure/{cell}/ncu_gpc"),
            "ncu_sm": _int(baseline.get("sm__cycles_active.max"), f"closure/{cell}/ncu_sm"),
            "ncu_inst": _int(baseline.get("smsp__inst_executed.sum"), f"closure/{cell}/ncu_inst"),
            "sim_completion": _int(simulator.get("completion_reference_cycles"), f"closure/{cell}/sim_completion"),
            "sm_gap": _int(vector.get("sm_completion_gap_cycles"), f"closure/{cell}/sm_gap"),
            "gpc_gap": _int(vector.get("gpc_completion_gap_cycles"), f"closure/{cell}/gpc_gap"),
            "launch_gap": _int(vector.get("gpc_launch_envelope_cycles"), f"closure/{cell}/launch_gap"),
            "dirty_writeback": tails.get("dirty_writeback") is True,
            "coherence": tails.get("coherence") is True,
        }
    if set(result) != set(EXPECTED_IDS):
        raise ValueError("closure report does not contain the exact six-cell cohort")
    return result


def _validate_summary(
    summary: Mapping[str, Any],
    closure: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    summary_path: Path,
    closure_path: Path,
    manifest_path: Path,
) -> None:
    expected = _closure_expected(closure)
    cohort = summary.get("cohort")
    if not isinstance(cohort, list) or len(cohort) != len(EXPECTED_IDS):
        raise ValueError("summary cohort must contain exactly six rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in cohort:
        if not isinstance(row, Mapping) or not isinstance(row.get("cell"), str):
            raise ValueError("summary cohort rows require an explicit cell id")
        by_id[str(row["cell"])] = row
    if set(by_id) != set(EXPECTED_IDS):
        raise ValueError("summary cohort ids differ from the six-cell closure")
    for cell, row in by_id.items():
        exp = expected[cell]
        closure_row = next(item for item in closure["rows"] if item["cell"] == cell)
        simulator = closure_row["simulator"]
        for summary_key, expected_key in (
            ("ncu_gpc_cycles", "ncu_gpc"),
            ("ncu_sm_active_cycles", "ncu_sm"),
            ("ncu_smsp_instructions", "ncu_inst"),
            ("sim_terminal_lower_bound_cycles", "sim_completion"),
        ):
            if _int(row.get(summary_key), f"summary/{cell}/{summary_key}") != exp[expected_key]:
                raise ValueError(f"summary/{cell}/{summary_key} disagrees with closure")
        if row.get("correctness") is not True:
            raise ValueError(f"summary/{cell}: correctness is not recorded as true")
        if _int(row.get("sim_instructions"), f"summary/{cell}/sim_instructions") != _int(simulator.get("instructions_issued"), f"closure/{cell}/instructions_issued"):
            raise ValueError(f"summary/{cell}/sim_instructions disagrees with closure")
        if row.get("frontier_gaps") != {
            "sm_completion_gap_cycles": exp["sm_gap"],
            "gpc_completion_gap_cycles": exp["gpc_gap"],
            "gpc_launch_envelope_cycles": exp["launch_gap"],
        }:
            raise ValueError(f"summary/{cell}/frontier_gaps disagrees with closure")
        if row.get("typed_tails") != simulator.get("tails"):
            raise ValueError(f"summary/{cell}/typed_tails disagrees with closure")
        manifest_cell = next(item for item in manifest["cells"] if item["id"] == cell)
        correctness_stage = manifest_cell["stages"]["correctness"]
        if row.get("correctness_binary") != correctness_stage.get("binary_path"):
            raise ValueError(f"summary/{cell}/correctness_binary disagrees with manifest")
        if row.get("correctness_binary_sha256") != correctness_stage.get("binary_sha256"):
            raise ValueError(f"summary/{cell}/correctness_binary_sha256 disagrees with manifest")
        if row.get("correctness_group_count") != correctness_stage.get("group_count"):
            raise ValueError(f"summary/{cell}/correctness_group_count disagrees with manifest")
    cp = by_id["cp_async_canary"]
    arguments = str(cp.get("arguments", ""))
    if (
        "matched stages (64 groups):" not in arguments
        or arguments.count("c11 l1 0 4 1") < 2
        or "separate 16,384-group correctness:" not in arguments
        or "separate 16,384-group NCU scaling:" not in arguments
    ):
        raise ValueError("summary/cp_async_canary: matched and scaling command variants are not explicit")
    if cp.get("async_matched_ncu_groups") != 64 or cp.get("async_scaling_observation_ncu_groups") != 16_384:
        raise ValueError("summary/cp_async_canary: group-count provenance is incomplete")
    correctness_capture = summary.get("correctness_capture")
    cp_manifest = next(item for item in manifest["cells"] if item["id"] == "cp_async_canary")
    if not isinstance(correctness_capture, Mapping):
        raise ValueError("summary correctness_capture is missing")
    matched_capture = correctness_capture.get("matched_cp_async")
    scaling_capture = correctness_capture.get("scaling_cp_async")
    if not isinstance(matched_capture, Mapping) or not isinstance(scaling_capture, Mapping):
        raise ValueError("summary must distinguish matched and scaling cp.async correctness")
    expected_matched = cp_manifest["stages"]["correctness"]
    expected_scaling = cp_manifest["scaling_observation"]["correctness"]
    for label, record, expected_stage, expected_groups in (
        ("matched_cp_async", matched_capture, expected_matched, 64),
        ("scaling_cp_async", scaling_capture, expected_scaling, 16_384),
    ):
        if (
            record.get("binary") != expected_stage.get("binary_path")
            or record.get("binary_sha256") != expected_stage.get("binary_sha256")
            or record.get("arguments") != expected_stage.get("argv", [])[1:]
            or record.get("group_count") != expected_groups
            or record.get("environment") != expected_stage.get("environment", {}).get("record")
        ):
            raise ValueError(f"summary correctness_capture/{label} is not bound to the manifest")
        for path_key in ("environment", "stdout", "stderr"):
            path_value = record.get(path_key)
            if not isinstance(path_value, str) or not (
                manifest_path.parent / path_value
            ).is_file():
                raise ValueError(
                    f"summary correctness_capture/{label}/{path_key} is missing"
                )
    core = summary.get("core_closure")
    analysis = closure.get("analysis")
    validation = analysis.get("input_validation") if isinstance(analysis, Mapping) else None
    if not isinstance(core, Mapping) or not isinstance(analysis, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("summary core_closure or closure analysis is missing")
    for key, expected_value in (
        ("decision", "no-go"),
        ("matched_comparable_rows", 6),
        ("retirement_tail_rows", 6),
        ("frontiers_remain_independent", True),
        ("aggregate_score_used", False),
    ):
        if core.get(key) != expected_value:
            raise ValueError(f"summary core_closure/{key} is stale")
    if validation.get("retirement_tail_rows") != 6 or validation.get("repeat_stable_rows") != 6:
        raise ValueError("summary core closure row counts disagree with closure analysis")
    if summary.get("closure_report_sha256") != _sha256(closure_path):
        raise ValueError("summary closure report hash is stale")
    if summary.get("campaign_manifest") != _relative_label(manifest_path):
        raise ValueError("summary campaign manifest path is stale")
    manifest_bundle = manifest.get("closure_analysis")
    if not isinstance(manifest_bundle, Mapping) or manifest_bundle.get("report_sha256") != _sha256(closure_path):
        raise ValueError("manifest closure report hash is stale")
    if summary.get("live_cohort_semantic_report"):
        live_report = REPO_ROOT / str(summary["live_cohort_semantic_report"])
        if live_report.is_file() and summary.get("live_cohort_semantic_report_sha256") != _sha256(live_report):
            raise ValueError("summary live cohort report hash is stale")
    del summary_path  # retained for a descriptive signature and future checks


def _relative_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    authority = manifest.get("runtime_authority")
    if (
        not isinstance(authority, Mapping)
        or not isinstance(authority.get("path"), str)
        or not isinstance(authority.get("sha256"), str)
        or not isinstance(authority.get("dso_closure"), Mapping)
        or authority.get("dso_count") != len(authority.get("dso_closure", {}))
    ):
        raise ValueError("manifest NVBit runtime authority is missing or malformed")
    authority_path = Path(str(authority["path"]))
    if not authority_path.is_file() or _sha256(authority_path) != authority["sha256"]:
        raise ValueError("manifest NVBit runtime authority hash is stale")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or [cell.get("id") for cell in cells if isinstance(cell, Mapping)] != list(EXPECTED_IDS):
        raise ValueError("manifest cells differ from the canonical six-cell order")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("manifest cell is malformed")
        cell_id = str(cell["id"])
        stages = cell.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != {"correctness", "ncu", "nvbit", "static_control", "simulator"}:
            raise ValueError(f"manifest/{cell_id}: stage set is incomplete")
        for stage_name, stage in stages.items():
            if not isinstance(stage, Mapping) or stage.get("timeout_seconds") != {"correctness": 180, "ncu": 900, "nvbit": 900, "static_control": 300, "simulator": 900}[stage_name]:
                raise ValueError(f"manifest/{cell_id}/{stage_name}: timeout is not explicit and canonical")
        if cell_id == "cp_async_canary":
            matched = cell.get("matched_execution")
            if not isinstance(matched, Mapping) or matched.get("stages") != ["correctness", "ncu", "nvbit", "static_control", "simulator"] or matched.get("group_count") != 64:
                raise ValueError("manifest/cp_async_canary: matched correctness stage is not declared")
            correctness = stages["correctness"]
            if correctness.get("binary_path") != cell["binary"].get("trace_path") or correctness.get("group_count") != 64:
                raise ValueError("manifest/cp_async_canary: correctness is not bound to the 64-group trace binary")
            required_limitation = (
                "Matched NCU and simulator stages both use 64 groups; only the separate "
                "16,384-group scaling observation uses 16,384 groups."
            )
            if required_limitation not in cell.get("limitations", []):
                raise ValueError("manifest/cp_async_canary: group-count limitation text is stale")


def validate(
    manifest_path: Path = DEFAULT_ROOT / "campaign_manifest.json",
    closure_path: Path = DEFAULT_ROOT / "closure_report.json",
    summary_path: Path = DEFAULT_ROOT / "summary.json",
    readme_path: Path = DEFAULT_ROOT / "README.md",
) -> None:
    manifest = _load(manifest_path)
    closure = _load(closure_path)
    summary = _load(summary_path)
    _validate_manifest(manifest)
    expected = _closure_expected(closure)
    actual = _parse_table(readme_path.read_text(encoding="utf-8"))
    if set(actual) != set(expected):
        raise ValueError("README residual table rows differ from closure")
    for cell in EXPECTED_IDS:
        if actual[cell] != expected[cell]:
            raise ValueError(f"README residual table/{cell} disagrees with closure")
    _validate_summary(
        summary,
        closure,
        manifest,
        summary_path=summary_path,
        closure_path=closure_path,
        manifest_path=manifest_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "campaign_manifest.json")
    parser.add_argument("--closure", type=Path, default=DEFAULT_ROOT / "closure_report.json")
    parser.add_argument("--summary", type=Path, default=DEFAULT_ROOT / "summary.json")
    parser.add_argument("--readme", type=Path, default=DEFAULT_ROOT / "README.md")
    args = parser.parse_args(argv)
    try:
        validate(args.manifest, args.closure, args.summary, args.readme)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SM86 handoff validation: {exc}", file=sys.stderr)
        return 1
    print("validated SM86 handoff: README residual table, summary, closure, and manifest agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
