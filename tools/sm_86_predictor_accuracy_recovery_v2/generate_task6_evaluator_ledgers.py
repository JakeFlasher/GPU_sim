"""Build task6 live-sweep evaluator ledgers.

The input sweep trees are produced by ``tools/local_labels/``. This
helper does not invoke NCU; it consolidates already-produced local-label
repeatability artifacts into compact admission and exclusion ledgers.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

DEFAULT_PREFLIGHT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "submission_candidate_preflight.json"
)
DEFAULT_READINESS = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "task6_submission_bundle_readiness_report.json"
)
DEFAULT_SWEEP_ROOTS = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "sweeps"
    / "sm86_submission_task6_round9",
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "sweeps"
    / "sm86_submission_task6_round9_recovery",
)
DEFAULT_OUT_SELECTED = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "task6_selected_label_admitted.json"
)
DEFAULT_OUT_CONFIG_MANIFEST = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "task6_selected_config_manifest.json"
)
DEFAULT_OUT_ADMITTED_LEDGER = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "task6_evaluator_admitted_ledger.json"
)
DEFAULT_OUT_EXCLUDED_LEDGER = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "task6_evaluator_excluded_ledger.json"
)
DEFAULT_OUT_SUMMARY = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "task6_live_sweep_summary.json"
)

STDDEV_THRESHOLD = 0.03
REQUIRED_ADMITTED_CELLS = 31
REQUIRED_UNIQUE_SLUGS = 30
REQUIRED_UNIT_METRICS = (
    "gpc__cycles_elapsed.max",
    "gpu__time_duration.sum",
)
REQUIRED_UNIT_METRIC_UNITS = {
    "gpc__cycles_elapsed.max": "cycles",
    "gpu__time_duration.sum": "ns",
}
PRODUCER_PROVENANCE_PATHS = (
    "tools/sm_86_predictor_accuracy_recovery_v2/"
    "generate_task6_evaluator_ledgers.py",
    "tools/local_labels/sweep_repeatability_matrix.py",
    "tools/local_labels/repeatability_aggregator.py",
)

VERDICT_COLUMNS = (
    "cell_id",
    "arch",
    "slug",
    "shape_point",
    "warmup_count",
    "cache_control",
    "repeats",
    "n_sidecars",
    "stddev_l2_sector",
    "stddev_l1tex_sector",
    "mean_l2_sector",
    "admitted",
    "detail",
)


def _utc_now() -> str:
    return (
        dt.datetime.now(tz=dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _rel(path: Path, repo_root: Path = _REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_contains_path(
    repo_root: Path,
    source_commit: str,
    relpath: str,
) -> bool:
    if source_commit in ("", "unknown"):
        return False
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e",
         f"{source_commit}:{relpath}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _assert_producer_provenance(
    repo_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Fail closed unless the artifact source commit contains the
    producer files whose behavior determines task6 admission.

    Unit tests that run outside a git repository receive
    ``source_commit='unknown'`` and skip the git-object check; real
    task6 artifacts are generated in-repo and must pass.
    """
    if source_commit == "unknown":
        return {
            "checked": False,
            "source_commit": source_commit,
            "producer_paths": list(PRODUCER_PROVENANCE_PATHS),
            "missing_paths": [],
        }
    missing = [
        relpath for relpath in PRODUCER_PROVENANCE_PATHS
        if not _git_commit_contains_path(repo_root, source_commit, relpath)
    ]
    if missing:
        raise ValueError(
            "task6 producer provenance is stale; source_commit "
            f"{source_commit} does not contain producer path(s): {missing!r}"
        )
    return {
        "checked": True,
        "source_commit": source_commit,
        "producer_paths": list(PRODUCER_PROVENANCE_PATHS),
        "missing_paths": [],
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _cell_key(arch: str, slug: str, shape_point: str) -> tuple[str, str, str]:
    return (str(arch), str(slug), str(shape_point))


def _cell_id(key: tuple[str, str, str]) -> str:
    return f"{key[0]}/{key[1]}/{key[2]}"


def _cell_dir(sub_sweep: Path, key: tuple[str, str, str]) -> Path:
    return sub_sweep / f"cell={key[0]}__{key[1]}__{key[2]}"


def _cfg_dir(
    sub_sweep: Path,
    key: tuple[str, str, str],
    warmup_count: int,
    cache_control: str,
) -> Path:
    return (
        _cell_dir(sub_sweep, key)
        / f"cfg=warmup{warmup_count}_cache_{cache_control}"
    )


def _read_verdict_rows(sub_sweep: Path) -> list[dict[str, str]]:
    csv_path = sub_sweep / "verdict-table.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing verdict table: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        missing = [col for col in VERDICT_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing columns {missing!r}")
        return list(reader)


def _bool_admitted(row: dict[str, str]) -> bool:
    return str(row.get("admitted", "")).strip().lower() == "true"


def _int_from_row(row: dict[str, str], key: str, fallback: int = 0) -> int:
    try:
        return int(str(row.get(key, "")).strip())
    except ValueError:
        return fallback


def _float_from_row(
    row: dict[str, str],
    key: str,
    fallback: float | None = None,
) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return fallback
    try:
        return float(str(raw))
    except ValueError:
        return fallback


def _stddev_key(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _read_cells(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = _load_json(path)
    cells = payload.get("cells", [])
    return list(cells) if isinstance(cells, list) else []


def _typed_reason_from_detail(detail: str) -> str:
    text = str(detail)
    lowered = text.lower()
    if "binary" in lowered and "missing" in lowered:
        return "binary_unavailable"
    if "no label_run_manifest" in lowered:
        return "label_provenance_unavailable"
    if "did not emit a cell record" in lowered:
        return "label_provenance_unavailable"
    if "aggregator summary not written" in lowered:
        return "label_repeatability_unavailable"
    if "returncode" in lowered:
        return "label_repeatability_unavailable"
    if "stddev" in lowered or "threshold" in lowered:
        return "label_repeatability_failed"
    return "label_repeatability_failed"


def _typed_reason_for_config_row(
    *,
    sub_sweep: Path,
    row: dict[str, str],
) -> str:
    key = _cell_key(row["arch"], row["slug"], row["shape_point"])
    warmup_count = _int_from_row(row, "warmup_count")
    cache_control = str(row.get("cache_control", ""))
    agg_dir = _cfg_dir(sub_sweep, key, warmup_count, cache_control) / (
        "_aggregator_out"
    )
    excluded = _read_cells(agg_dir / "label_excluded.json")
    for cell in excluded:
        if (
            str(cell.get("slug")) == key[1]
            and str(cell.get("shape_point")) == key[2]
        ):
            reason = str(cell.get("typed_reason", "")).strip()
            if reason:
                return reason
    repeatability = _load_json(agg_dir / "label_repeatability.json") if (
        agg_dir / "label_repeatability.json"
    ).is_file() else {}
    for cell in repeatability.get("cells", []) or []:
        if (
            str(cell.get("slug")) == key[1]
            and str(cell.get("shape_point")) == key[2]
        ):
            admission = cell.get("admission", {}) or {}
            reason = str(admission.get("typed_reason", "")).strip()
            if reason:
                return reason
    return _typed_reason_from_detail(str(row.get("detail", "")))


def _readiness_index(readiness: dict[str, Any]) -> dict[tuple[str, str, str], list[dict]]:
    out: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    for row in readiness.get("rows", []) or []:
        cell = row.get("cell", {}) or {}
        key = _cell_key(
            str(cell.get("arch", "")),
            str(cell.get("slug", "")),
            str(cell.get("shape_point", "")),
        )
        if all(key):
            out[key].append(row)
    return out


def _readiness_context(
    rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not rows:
        return {
            "matching_row_count": 0,
            "classifications": [],
            "typed_reasons": [],
        }
    classifications = sorted({str(r.get("classification", "")) for r in rows})
    typed_reasons = sorted({
        str(reason)
        for row in rows
        for reason in (row.get("typed_reasons", []) or [])
        if str(reason)
    })
    ready_paths = sorted({
        str(((row.get("bundle_readiness", {}) or {}).get("ready_path")))
        for row in rows
        if ((row.get("bundle_readiness", {}) or {}).get("ready_path"))
    })
    return {
        "matching_row_count": len(rows),
        "classifications": classifications,
        "typed_reasons": typed_reasons,
        "ready_bundle_paths": ready_paths,
    }


def _unit_metrics_by_id(cell: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in cell.get("unit_tagged_metrics", []) or []:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("ncu_metric", ""))
        if metric:
            out[metric] = row
    return out


def _required_unit_metric_values(cell: dict[str, Any]) -> dict[str, Any]:
    by_metric = _unit_metrics_by_id(cell)
    return {
        metric: by_metric.get(metric, {}).get("value")
        for metric in REQUIRED_UNIT_METRICS
    }


def _missing_required_label_reasons(cell: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if cell.get("cache_hit_rate_l2_sector_pct") is None:
        missing.append("missing_l2_sector_hit_rate")
    if cell.get("cache_hit_rate_l1tex_sector_pct") is None:
        missing.append("missing_l1tex_sector_hit_rate")
    by_metric = _unit_metrics_by_id(cell)
    for metric in REQUIRED_UNIT_METRICS:
        row = by_metric.get(metric)
        if row is None:
            missing.append(f"missing_unit_metric:{metric}")
            continue
        if row.get("value") is None:
            missing.append(f"missing_unit_metric_value:{metric}")
        expected_unit = REQUIRED_UNIT_METRIC_UNITS[metric]
        if row.get("unit") != expected_unit:
            missing.append(
                f"unit_metric_unit_mismatch:{metric}:"
                f"{row.get('unit')}!={expected_unit}"
            )
    return missing


def _selected_bundle_blocker_reasons(
    rows: list[dict[str, Any]] | None,
) -> list[str]:
    context = _readiness_context(rows)
    reasons = list(context["typed_reasons"])
    if context["matching_row_count"] == 0:
        reasons.append("missing_bundle_readiness_row")
    if "bundle_ready" not in set(context["classifications"]):
        reasons.append("selected_bundle_not_ready")
    return sorted(set(str(reason) for reason in reasons if str(reason)))


def _partition_publication_slice(
    *,
    selected_cells: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    readiness_by_key: dict[tuple[str, str, str], list[dict]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    admitted_cells: list[dict[str, Any]] = []
    admitted_manifest_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for cell, manifest_row in zip(selected_cells, manifest_rows):
        key = _cell_key(cell["arch"], cell["slug"], cell["shape_point"])
        missing_labels = _missing_required_label_reasons(cell)
        bundle_blockers = _selected_bundle_blocker_reasons(
            readiness_by_key.get(key)
        )
        reasons = sorted(set(missing_labels + bundle_blockers))
        if reasons:
            excluded_rows.append({
                "cell_id": _cell_id(key),
                "arch": key[0],
                "slug": key[1],
                "shape_point": key[2],
                "typed_reason": reasons[0],
                "typed_reasons": reasons,
                "detail": (
                    "selected candidate removed from the task6 "
                    "publication slice because required labels or "
                    "bundle-readiness gates failed"
                ),
                "selected_config": dict(manifest_row),
            })
            continue
        admitted_cells.append(cell)
        admitted_manifest_rows.append(manifest_row)
    return admitted_cells, admitted_manifest_rows, excluded_rows


def _live_ready_index(
    preflight: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, Any]]],
    dict[str, int],
]:
    rows = [
        row for row in preflight.get("rows", []) or []
        if row.get("preflight_status") == "live_ready"
    ]
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        by_key[_cell_key(row["arch"], row["slug"], row["shape_point"])].append(row)
    summary = {
        "live_ready_rows": len(rows),
        "live_ready_unique_cells": len(by_key),
        "live_ready_unique_slugs": len({key[1] for key in by_key}),
        "duplicate_live_ready_rows": len(rows) - len(by_key),
    }
    return by_key, summary


def _config_row_record(
    *,
    sub_sweep: Path,
    row: dict[str, str],
    typed_reason: str,
) -> dict[str, Any]:
    return {
        "cell_id": row["cell_id"],
        "arch": row["arch"],
        "slug": row["slug"],
        "shape_point": row["shape_point"],
        "warmup_count": _int_from_row(row, "warmup_count"),
        "cache_control": row["cache_control"],
        "repeats": _int_from_row(row, "repeats"),
        "n_sidecars": _int_from_row(row, "n_sidecars"),
        "stddev_l2_sector": _float_from_row(row, "stddev_l2_sector"),
        "stddev_l1tex_sector": _float_from_row(row, "stddev_l1tex_sector"),
        "mean_l2_sector": _float_from_row(row, "mean_l2_sector"),
        "typed_reason": typed_reason,
        "detail": str(row.get("detail", "")),
        "sub_sweep_relpath": _rel(sub_sweep),
    }


def _collect_failed_invocations(sub_sweep: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(sub_sweep.rglob("failed_invocation_*.json")):
        payload = _load_json(path)
        records.append({
            "cell_id": (
                f"{payload.get('arch')}/{payload.get('slug')}/"
                f"{payload.get('shape_point')}"
            ),
            "arch": payload.get("arch"),
            "slug": payload.get("slug"),
            "shape_point": payload.get("shape_point"),
            "warmup_count": int(payload.get("warmup_count", 0)),
            "cache_control": payload.get("cache_control"),
            "repeat_index": int(payload.get("repeat_index", 0)),
            "launch_index": int(payload.get("launch_index", 0)),
            "label_driver_returncode": int(
                payload.get("label_driver_returncode", 0)
            ),
            "stderr_excerpt": payload.get("label_driver_stderr_excerpt", ""),
            "typed_reason": "label_invocation_failed",
            "relpath": _rel(path),
            "sub_sweep_relpath": _rel(sub_sweep),
        })
    return records


def _group_failed_invocations(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for record in records:
        grouped[
            (
                str(record["cell_id"]),
                int(record["warmup_count"]),
                str(record["cache_control"]),
            )
        ].append(record)
    out: list[dict[str, Any]] = []
    for (cell_id, warmup_count, cache_control), items in sorted(grouped.items()):
        out.append({
            "cell_id": cell_id,
            "warmup_count": warmup_count,
            "cache_control": cache_control,
            "failed_invocation_count": len(items),
            "returncodes": sorted({
                int(item["label_driver_returncode"]) for item in items
            }),
            "typed_reason": "label_invocation_failed",
            "relpaths": [item["relpath"] for item in items],
        })
    return out


def _collect_sweep(
    sub_sweep: Path,
) -> dict[str, Any]:
    rows = _read_verdict_rows(sub_sweep)
    candidates: list[dict[str, Any]] = []
    non_admitted_rows: list[dict[str, Any]] = []
    inconsistent_rows: list[dict[str, Any]] = []

    for row in rows:
        key = _cell_key(row["arch"], row["slug"], row["shape_point"])
        warmup_count = _int_from_row(row, "warmup_count")
        cache_control = row["cache_control"]
        agg_path = (
            _cfg_dir(sub_sweep, key, warmup_count, cache_control)
            / "_aggregator_out"
            / "label_admitted.json"
        )
        if not _bool_admitted(row):
            non_admitted_rows.append(_config_row_record(
                sub_sweep=sub_sweep,
                row=row,
                typed_reason=_typed_reason_for_config_row(
                    sub_sweep=sub_sweep,
                    row=row,
                ),
            ))
            continue
        cells = _read_cells(agg_path)
        matching_cells = [
            cell for cell in cells
            if (
                str(cell.get("arch")) == key[0]
                and str(cell.get("slug")) == key[1]
                and str(cell.get("shape_point")) == key[2]
            )
        ]
        if not matching_cells:
            inconsistent_rows.append({
                "cell_id": row["cell_id"],
                "arch": row["arch"],
                "slug": row["slug"],
                "shape_point": row["shape_point"],
                "warmup_count": warmup_count,
                "cache_control": cache_control,
                "repeats": _int_from_row(row, "repeats"),
                "n_sidecars": _int_from_row(row, "n_sidecars"),
                "typed_reason": "label_payload_unavailable",
                "detail": (
                    "verdict row was admitted but the matching "
                    "label_admitted payload was empty or absent"
                ),
                "sub_sweep_relpath": _rel(sub_sweep),
                "source_relpath": _rel(agg_path, sub_sweep),
            })
            continue
        source_relpath = _rel(agg_path, sub_sweep)
        for cell in matching_cells:
            candidates.append({
                "cell": dict(cell),
                "warmup_count": warmup_count,
                "cache_control": cache_control,
                "source_relpath": source_relpath,
                "source_sweep_relpath": _rel(sub_sweep),
                "source_sha256": _sha256(agg_path),
            })

    failed = _collect_failed_invocations(sub_sweep)
    verdict = _load_json(sub_sweep / "verdict.json") if (
        sub_sweep / "verdict.json"
    ).is_file() else {}
    summary = {
        "sub_sweep_relpath": _rel(sub_sweep),
        "verdict": verdict,
        "verdict_table_sha256": _sha256(sub_sweep / "verdict-table.csv"),
        "verdict_table_json_sha256": _sha256(sub_sweep / "verdict-table.json"),
        "config_rows": len(rows),
        "unique_cells": len({
            _cell_key(row["arch"], row["slug"], row["shape_point"])
            for row in rows
        }),
        "admitted_config_rows": sum(1 for row in rows if _bool_admitted(row)),
        "non_admitted_config_rows": len(non_admitted_rows),
        "admitted_config_rows_with_missing_payload": len(inconsistent_rows),
        "label_sidecars": len(list(sub_sweep.rglob("label_[0-9]*_[0-9]*.json"))),
        "failed_invocations": len(failed),
        "repeatability_summaries": len(list(sub_sweep.rglob("label_repeatability.json"))),
        "planned_invocations": sum(_int_from_row(row, "repeats") for row in rows),
    }
    return {
        "summary": summary,
        "candidates": candidates,
        "non_admitted_config_rows": non_admitted_rows,
        "inconsistent_admitted_config_rows": inconsistent_rows,
        "failed_invocations": failed,
    }


def _select_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for candidate in candidates:
        cell = candidate["cell"]
        key = _cell_key(cell["arch"], cell["slug"], cell["shape_point"])
        grouped[key].append(candidate)

    selected_cells: list[dict[str, Any]] = []
    selected_manifest_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        options = grouped[key]
        options.sort(
            key=lambda item: (
                len(_missing_required_label_reasons(item["cell"])),
                _stddev_key(item["cell"].get("stddev_l2_sector")),
                _stddev_key(item["cell"].get("stddev_l1tex_sector")),
                int(item["warmup_count"]),
                str(item["cache_control"]),
                str(item["source_sweep_relpath"]),
                str(item["source_relpath"]),
            )
        )
        chosen = options[0]
        cell = dict(chosen["cell"])
        selected_cells.append(cell)
        selected_manifest_rows.append({
            "arch": key[0],
            "slug": key[1],
            "shape_point": key[2],
            "warmup_count": int(chosen["warmup_count"]),
            "cache_control": str(chosen["cache_control"]),
            "source_sweep_relpath": str(chosen["source_sweep_relpath"]),
            "source_relpath": str(chosen["source_relpath"]),
            "source_sha256": chosen["source_sha256"],
            "stddev_l2_sector": cell.get("stddev_l2_sector"),
            "stddev_l1tex_sector": cell.get("stddev_l1tex_sector"),
        })
    return selected_cells, selected_manifest_rows


def _selected_ledger_rows(
    *,
    selected_cells: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    preflight_by_key: dict[tuple[str, str, str], list[dict[str, Any]]],
    readiness_by_key: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    manifest_by_key = {
        _cell_key(row["arch"], row["slug"], row["shape_point"]): row
        for row in manifest_rows
    }
    out: list[dict[str, Any]] = []
    for cell in selected_cells:
        key = _cell_key(cell["arch"], cell["slug"], cell["shape_point"])
        preflight_rows = preflight_by_key.get(key, [])
        preflight_context = {
            "matching_row_count": len(preflight_rows),
            "family_buckets": sorted({
                str(row.get("family_bucket", "")) for row in preflight_rows
                if row.get("family_bucket")
            }),
            "roles": sorted({
                str(row.get("role", "")) for row in preflight_rows
                if row.get("role")
            }),
            "source_commits": sorted({
                str(row.get("source_commit", "")) for row in preflight_rows
                if row.get("source_commit")
            }),
            "binary_sha256s": sorted({
                str(row.get("binary_sha256", "")) for row in preflight_rows
                if row.get("binary_sha256")
            }),
            "predictor_trace_paths": sorted({
                str(row.get("expected_predictor_trace_path", ""))
                for row in preflight_rows
                if row.get("expected_predictor_trace_path")
            }),
        }
        out.append({
            "cell_id": _cell_id(key),
            "cell": {
                "arch": key[0],
                "slug": key[1],
                "shape_point": key[2],
            },
            "label_identity": {
                "binary_sha256": cell.get("binary_sha256"),
                "kernel_signature_sha256": cell.get(
                    "kernel_signature_sha256"
                ),
                "source_commit": cell.get("source_commit"),
            },
            "repeatability": {
                "n_repeats": cell.get("n_repeats"),
                "stddev_l2_sector": cell.get("stddev_l2_sector"),
                "stddev_l1tex_sector": cell.get("stddev_l1tex_sector"),
                "stddev_threshold": STDDEV_THRESHOLD,
                "cache_hit_rate_l2_sector_pct": cell.get(
                    "cache_hit_rate_l2_sector_pct"
                ),
                "cache_hit_rate_l1tex_sector_pct": cell.get(
                    "cache_hit_rate_l1tex_sector_pct"
                ),
                "l1tex_label_status": cell.get("l1tex_label_status"),
                "unit_tagged_metrics": cell.get("unit_tagged_metrics", []),
                "required_unit_metric_values": _required_unit_metric_values(
                    cell
                ),
            },
            "selected_config": manifest_by_key[key],
            "preflight_context": preflight_context,
            "bundle_readiness_context": _readiness_context(
                readiness_by_key.get(key),
            ),
        })
    return out


def _final_excluded_cells(
    *,
    live_ready_keys: set[tuple[str, str, str]],
    selected_keys: set[tuple[str, str, str]],
    non_admitted_rows: list[dict[str, Any]],
    inconsistent_rows: list[dict[str, Any]],
    selected_excluded_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reasons_by_key: dict[tuple[str, str, str], set[str]] = (
        collections.defaultdict(set)
    )
    details_by_key: dict[tuple[str, str, str], list[str]] = (
        collections.defaultdict(list)
    )
    for row in non_admitted_rows + inconsistent_rows + selected_excluded_rows:
        key = _cell_key(row["arch"], row["slug"], row["shape_point"])
        typed_reasons = row.get("typed_reasons")
        if isinstance(typed_reasons, list):
            for reason in typed_reasons:
                reasons_by_key[key].add(str(reason))
        else:
            reasons_by_key[key].add(str(row["typed_reason"]))
        detail = str(row.get("detail", ""))
        if detail:
            details_by_key[key].append(detail)
    out: list[dict[str, Any]] = []
    for key in sorted(live_ready_keys - selected_keys):
        reasons = sorted(reasons_by_key.get(key, {"label_not_admitted"}))
        out.append({
            "cell_id": _cell_id(key),
            "cell": {
                "arch": key[0],
                "slug": key[1],
                "shape_point": key[2],
            },
            "typed_reasons": reasons,
            "details": sorted(set(details_by_key.get(key, []))),
        })
    return out


def _protocol_from_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cache_controls": sorted({str(row["cache_control"]) for row in rows}),
        "warmup_counts": sorted({int(row["warmup_count"]) for row in rows}),
        "repeats": sorted({int(row["repeats"]) for row in rows}),
        "stddev_threshold": STDDEV_THRESHOLD,
    }


def _discover_default_sub_sweeps() -> list[Path]:
    out: list[Path] = []
    for root in DEFAULT_SWEEP_ROOTS:
        if not root.is_dir():
            continue
        children = sorted(path for path in root.glob("sweep_*") if path.is_dir())
        if children:
            out.append(children[-1])
    return out


def build_task6_ledgers(
    *,
    preflight_path: Path,
    readiness_path: Path,
    sub_sweeps: list[Path],
    out_selected_path: Path,
    out_config_manifest_path: Path,
    out_admitted_ledger_path: Path,
    out_excluded_ledger_path: Path,
    out_summary_path: Path,
    repo_root: Path = _REPO_ROOT,
    required_admitted_cells: int = REQUIRED_ADMITTED_CELLS,
    required_unique_slugs: int = REQUIRED_UNIQUE_SLUGS,
) -> dict[str, dict[str, Any]]:
    if not sub_sweeps:
        raise ValueError("at least one sub-sweep is required")
    generated_at = _utc_now()
    source_commit = _git_head(repo_root)
    producer_provenance = _assert_producer_provenance(repo_root, source_commit)
    preflight = _load_json(preflight_path)
    readiness = _load_json(readiness_path)
    preflight_by_key, preflight_summary = _live_ready_index(preflight)
    readiness_by_key = _readiness_index(readiness)

    collected = [_collect_sweep(path) for path in sub_sweeps]
    candidates = [
        candidate
        for item in collected
        for candidate in item["candidates"]
    ]
    non_admitted_rows = [
        row
        for item in collected
        for row in item["non_admitted_config_rows"]
    ]
    inconsistent_rows = [
        row
        for item in collected
        for row in item["inconsistent_admitted_config_rows"]
    ]
    failed_invocations = [
        row
        for item in collected
        for row in item["failed_invocations"]
    ]
    selected_cells, manifest_rows = _select_candidates(candidates)
    selected_cells, manifest_rows, selected_excluded_rows = (
        _partition_publication_slice(
            selected_cells=selected_cells,
            manifest_rows=manifest_rows,
            readiness_by_key=readiness_by_key,
        )
    )
    selected_keys = {
        _cell_key(cell["arch"], cell["slug"], cell["shape_point"])
        for cell in selected_cells
    }
    final_excluded = _final_excluded_cells(
        live_ready_keys=set(preflight_by_key),
        selected_keys=selected_keys,
        non_admitted_rows=non_admitted_rows,
        inconsistent_rows=inconsistent_rows,
        selected_excluded_rows=selected_excluded_rows,
    )

    admitted_cell_count = len(selected_cells)
    admitted_unique_slug_count = len({cell["slug"] for cell in selected_cells})
    floor_check = {
        "required_admitted_cells": int(required_admitted_cells),
        "required_unique_slugs": int(required_unique_slugs),
        "admitted_cell_count": admitted_cell_count,
        "admitted_unique_slug_count": admitted_unique_slug_count,
        "passes": (
            admitted_cell_count >= required_admitted_cells
            and admitted_unique_slug_count >= required_unique_slugs
        ),
    }
    if not floor_check["passes"]:
        raise ValueError(
            "task6 selected labels do not meet the required floors: "
            f"{floor_check}; selected_excluded="
            + json.dumps(selected_excluded_rows[:12], sort_keys=True)
        )

    protocol = _protocol_from_rows(
        non_admitted_rows
        + [
            {
                "warmup_count": row["warmup_count"],
                "cache_control": row["cache_control"],
                "repeats": row["repeats"],
            }
            for item in collected
            for row in _read_verdict_rows(
                repo_root / item["summary"]["sub_sweep_relpath"]
            )
        ]
    )

    selected_payload = {
        "schema_version": 5,
        "cells": selected_cells,
    }
    selected_manifest = {
        "artifact_kind": "task6_selected_config_manifest",
        "generated_at_utc": generated_at,
        "schema_version": 1,
        "source_commit": source_commit,
        "selected_per_cell": manifest_rows,
        "sub_sweep_relpaths": [_rel(path, repo_root) for path in sub_sweeps],
    }
    admitted_ledger = {
        "artifact_kind": "task6_evaluator_admitted_ledger",
        "generated_at_utc": generated_at,
        "schema_version": 1,
        "source_commit": source_commit,
        "summary": {
            **preflight_summary,
            "admitted_cell_count": admitted_cell_count,
            "admitted_unique_slug_count": admitted_unique_slug_count,
            "final_excluded_cell_count": len(final_excluded),
            "floor_check": floor_check,
            "protocol": protocol,
            "required_unit_metrics": list(REQUIRED_UNIT_METRICS),
            "producer_provenance": producer_provenance,
        },
        "admitted_cells": _selected_ledger_rows(
            selected_cells=selected_cells,
            manifest_rows=manifest_rows,
            preflight_by_key=preflight_by_key,
            readiness_by_key=readiness_by_key,
        ),
    }
    excluded_ledger = {
        "artifact_kind": "task6_evaluator_excluded_ledger",
        "generated_at_utc": generated_at,
        "schema_version": 1,
        "source_commit": source_commit,
        "summary": {
            **preflight_summary,
            "final_excluded_cell_count": len(final_excluded),
            "non_admitted_config_row_count": len(non_admitted_rows),
            "admitted_config_rows_with_missing_payload": len(inconsistent_rows),
            "selected_candidates_removed_by_publication_gates": len(
                selected_excluded_rows
            ),
            "failed_invocation_count": len(failed_invocations),
            "required_unit_metrics": list(REQUIRED_UNIT_METRICS),
            "producer_provenance": producer_provenance,
            "no_untyped_non_admitted_config_rows": all(
                bool(row.get("typed_reason"))
                for row in non_admitted_rows
            ),
            "no_untyped_inconsistent_admitted_config_rows": all(
                bool(row.get("typed_reason"))
                for row in inconsistent_rows
            ),
        },
        "final_excluded_cells": final_excluded,
        "non_admitted_config_rows": non_admitted_rows,
        "inconsistent_admitted_config_rows": inconsistent_rows,
        "selected_candidates_removed_by_publication_gates": (
            selected_excluded_rows
        ),
        "failed_invocation_groups": _group_failed_invocations(
            failed_invocations
        ),
    }
    sweep_summary = {
        "artifact_kind": "task6_live_sweep_summary",
        "generated_at_utc": generated_at,
        "schema_version": 1,
        "source_commit": source_commit,
        "summary": {
            **preflight_summary,
            "sweep_count": len(sub_sweeps),
            "config_rows": sum(
                item["summary"]["config_rows"] for item in collected
            ),
            "planned_invocations": sum(
                item["summary"]["planned_invocations"] for item in collected
            ),
            "label_sidecars": sum(
                item["summary"]["label_sidecars"] for item in collected
            ),
            "failed_invocations": len(failed_invocations),
            "repeatability_summaries": sum(
                item["summary"]["repeatability_summaries"]
                for item in collected
            ),
            "admitted_config_rows": sum(
                item["summary"]["admitted_config_rows"]
                for item in collected
            ),
            "non_admitted_config_rows": len(non_admitted_rows),
            "admitted_config_rows_with_missing_payload": len(inconsistent_rows),
            "selected_candidates_removed_by_publication_gates": len(
                selected_excluded_rows
            ),
            "selected_admitted_cells": admitted_cell_count,
            "selected_unique_slugs": admitted_unique_slug_count,
            "final_excluded_cells": len(final_excluded),
            "floor_check": floor_check,
            "protocol": protocol,
            "required_unit_metrics": list(REQUIRED_UNIT_METRICS),
            "producer_provenance": producer_provenance,
        },
        "sweeps": [item["summary"] for item in collected],
    }

    _write_json(out_selected_path, selected_payload)
    _write_json(out_config_manifest_path, selected_manifest)
    _write_json(out_admitted_ledger_path, admitted_ledger)
    _write_json(out_excluded_ledger_path, excluded_ledger)
    _write_json(out_summary_path, sweep_summary)
    return {
        "selected_label_admitted": selected_payload,
        "selected_config_manifest": selected_manifest,
        "admitted_ledger": admitted_ledger,
        "excluded_ledger": excluded_ledger,
        "live_sweep_summary": sweep_summary,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate task6 evaluator ledgers from local-label sweeps.",
    )
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument(
        "--sub-sweep",
        action="append",
        type=Path,
        default=None,
        help="Repeatable path to a sweep_* directory.",
    )
    parser.add_argument(
        "--out-selected",
        type=Path,
        default=DEFAULT_OUT_SELECTED,
    )
    parser.add_argument(
        "--out-config-manifest",
        type=Path,
        default=DEFAULT_OUT_CONFIG_MANIFEST,
    )
    parser.add_argument(
        "--out-admitted-ledger",
        type=Path,
        default=DEFAULT_OUT_ADMITTED_LEDGER,
    )
    parser.add_argument(
        "--out-excluded-ledger",
        type=Path,
        default=DEFAULT_OUT_EXCLUDED_LEDGER,
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=DEFAULT_OUT_SUMMARY,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    sub_sweeps = args.sub_sweep if args.sub_sweep else _discover_default_sub_sweeps()
    try:
        outputs = build_task6_ledgers(
            preflight_path=args.preflight,
            readiness_path=args.readiness,
            sub_sweeps=list(sub_sweeps),
            out_selected_path=args.out_selected,
            out_config_manifest_path=args.out_config_manifest,
            out_admitted_ledger_path=args.out_admitted_ledger,
            out_excluded_ledger_path=args.out_excluded_ledger,
            out_summary_path=args.out_summary,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"generate_task6_evaluator_ledgers: {exc}", file=sys.stderr)
        return 1

    summary = outputs["live_sweep_summary"]["summary"]
    print(
        "generate_task6_evaluator_ledgers: "
        f"selected={summary['selected_admitted_cells']} "
        f"unique_slugs={summary['selected_unique_slugs']} "
        f"final_excluded={summary['final_excluded_cells']}"
    )
    print(f"  admitted ledger: {_rel(args.out_admitted_ledger)}")
    print(f"  excluded ledger: {_rel(args.out_excluded_ledger)}")
    print(f"  summary: {_rel(args.out_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
