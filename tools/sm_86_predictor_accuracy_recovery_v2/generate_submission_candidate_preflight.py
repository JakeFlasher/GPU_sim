"""Generate the sm_86 submission candidate preflight artifact.

The input roster is a planning artifact. This tool turns each
compatible roster row into a closed pass/fail live-run preflight row:
binary path and hash, predictor trace JSON, measurement manifest,
manifest-vs-binary validation, provenance audit, and NCU metric
availability confirmation.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.local_labels import provenance_audit  # noqa: E402
from tools.local_labels.stage_measurement_manifests_from_predictor_trace import (  # noqa: E402
    _build_full_manifest_from_trace,
    _existing_manifest_is_rich,
    _merge_join_keys_into_existing,
)
from tools.local_labels.validate_measurement_manifest import (  # noqa: E402
    _check_binary_map_coverage,
    _compute_binary_sha256,
    _validate_schema,
)


DEFAULT_ROSTER = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "submission_expansion_roster.json"
)
DEFAULT_OUT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "submission_candidate_preflight.json"
)
DEFAULT_METRIC_AVAILABILITY = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "ncu_metric_availability_probe_sm_86.json"
)

_SUPPORTED_WORKLOAD_PREFIXES = (
    ("workloads/portable_lcd/", "portable_lcd"),
    ("workloads/simbench_bw/", "simbench_bw"),
)

AuditRunner = Callable[..., provenance_audit.AuditResult]


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return (
        dt.datetime.now(tz=dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_sha256(workload_root: Path, slug: str) -> str | None:
    source = workload_root / f"{slug}.cu"
    if not source.is_file():
        return None
    return _sha256(source)


def _workload_source(row: dict[str, Any]) -> str:
    root = str(row["workload_root"])
    for prefix, source in _SUPPORTED_WORKLOAD_PREFIXES:
        if root.startswith(prefix):
            return source
    raise ValueError(f"unknown workload_root prefix for {root!r}")


def _cell_root(
    *, repo_root: Path, row: dict[str, Any], sku: str, arch_dir: str
) -> Path:
    source = _workload_source(row)
    return (
        repo_root
        / "workloads"
        / source
        / "results"
        / str(row["slug"])
        / sku
        / arch_dir
    )


def _binary_path(repo_root: Path, row: dict[str, Any]) -> Path:
    return repo_root / str(row["workload_root"]) / f"{row['slug']}_run"


def _trace_path(
    *, repo_root: Path, row: dict[str, Any], sku: str, arch_dir: str
) -> Path:
    return _cell_root(repo_root=repo_root, row=row, sku=sku, arch_dir=arch_dir) / (
        f"trace_{row['shape_point']}.json"
    )


def _manifest_path(
    *, repo_root: Path, row: dict[str, Any], sku: str, arch_dir: str
) -> Path:
    return _cell_root(repo_root=repo_root, row=row, sku=sku, arch_dir=arch_dir) / (
        f"measurement_manifest_{row['shape_point']}.json"
    )


def _load_metric_availability(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "missing",
            "path": _rel(path, repo_root),
            "confirmed": False,
            "ncu_metric_version": None,
            "n_targets": 0,
            "n_available": 0,
            "n_excluded": 0,
            "typed_exclusions": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    typed_exclusions = [
        {
            "ncu_metric": metric.get("ncu_metric"),
            "excluded_reason": metric.get("excluded_reason"),
        }
        for metric in payload.get("metrics", [])
        if not metric.get("available")
    ]
    confirmed = bool(
        payload.get("ncu_metric_version")
        and payload.get("n_targets", 0) > 0
        and payload.get("n_available", 0) > 0
    )
    return {
        "status": "confirmed" if confirmed else "unconfirmed",
        "path": _rel(path, repo_root),
        "confirmed": confirmed,
        "arch": payload.get("arch"),
        "device": payload.get("device"),
        "ncu_metric_version": payload.get("ncu_metric_version"),
        "n_targets": payload.get("n_targets", 0),
        "n_available": payload.get("n_available", 0),
        "n_excluded": payload.get("n_excluded", 0),
        "typed_exclusions": typed_exclusions,
    }


def _stage_manifest_from_trace(
    *, trace_path: Path, manifest_path: Path, sku: str, shape_point: str
) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for key in (
        "slug",
        "shape_point",
        "binary_sha256",
        "kernel_signature_sha256",
    ):
        if not trace.get(key):
            raise ValueError(f"trace_missing_join_key:{key}")
    if str(trace["shape_point"]) != shape_point:
        raise ValueError(
            "trace_shape_point_mismatch:"
            f"trace={trace['shape_point']!r}:expected={shape_point!r}"
        )
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _existing_manifest_is_rich(existing):
            manifest = _merge_join_keys_into_existing(
                existing=existing, trace=trace, shape_point=shape_point
            )
        else:
            manifest = _build_full_manifest_from_trace(
                trace=trace, sku=sku, shape_point=shape_point
            )
    else:
        manifest = _build_full_manifest_from_trace(
            trace=trace, sku=sku, shape_point=shape_point
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_manifest_against_binary(
    *,
    manifest_path: Path,
    binary_path: Path,
    arch_dir: str,
) -> tuple[bool, str | None, str | None]:
    if not manifest_path.is_file():
        return False, "manifest_missing", f"manifest not found at {manifest_path}"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, "manifest_json_parse_error", str(exc)
    if not isinstance(payload, dict):
        return False, "manifest_schema_invalid", "manifest root is not an object"

    schema_error = _validate_schema(payload, manifest_path)
    if schema_error is not None:
        return False, "manifest_schema_invalid", schema_error

    cell_key = f"{arch_dir}/{payload['slug']}/{payload['shape_point']}"
    coverage_error = _check_binary_map_coverage(
        [(manifest_path, payload)], {cell_key: binary_path}
    )
    if coverage_error is not None:
        return False, "manifest_binary_coverage_mismatch", coverage_error

    if not binary_path.is_file():
        return False, "binary_missing", f"binary not found at {binary_path}"
    try:
        actual_sha = _compute_binary_sha256(binary_path)
    except OSError as exc:
        return False, "binary_hash_io_error", str(exc)
    if actual_sha != payload["binary_sha256"]:
        return (
            False,
            "manifest_binary_sha256_mismatch",
            f"manifest={payload['binary_sha256']} actual={actual_sha}",
        )
    return True, None, None


def _trace_identity_failure(
    *, row: dict[str, Any], trace_payload: dict[str, Any]
) -> str | None:
    if str(trace_payload.get("slug")) != str(row["slug"]):
        return (
            "predictor_trace_slug_mismatch:"
            f"trace={trace_payload.get('slug')!r}:expected={row['slug']!r}"
        )
    if str(trace_payload.get("shape_point")) != str(row["shape_point"]):
        return (
            "predictor_trace_shape_point_mismatch:"
            f"trace={trace_payload.get('shape_point')!r}:"
            f"expected={row['shape_point']!r}"
        )
    return None


def _preflight_row(
    *,
    row: dict[str, Any],
    repo_root: Path,
    sku: str,
    arch_dir: str,
    metric_availability: dict[str, Any],
    audit_runner: AuditRunner,
    live_sweep: bool,
) -> dict[str, Any]:
    slug = str(row["slug"])
    shape_point = str(row["shape_point"])
    workload_root = repo_root / str(row["workload_root"])
    binary_path = _binary_path(repo_root, row)
    trace_path = _trace_path(
        repo_root=repo_root, row=row, sku=sku, arch_dir=arch_dir
    )
    manifest_path = _manifest_path(
        repo_root=repo_root, row=row, sku=sku, arch_dir=arch_dir
    )

    entry: dict[str, Any] = {
        "role": row["role"],
        "slug": slug,
        "shape_point": shape_point,
        "family_bucket": row["family_bucket"],
        "workload_source": _workload_source(row),
        "workload_root": row["workload_root"],
        "arch": arch_dir,
        "gpu_sku": sku,
        "binary_path": _rel(binary_path, repo_root),
        "binary_sha256": None,
        "source_kernel_signature_sha256": _source_sha256(workload_root, slug),
        "expected_predictor_trace_path": _rel(trace_path, repo_root),
        "expected_manifest_path": _rel(manifest_path, repo_root),
        "staged_manifest_path": None,
        "source_commit": None,
        "kernel_signature_sha256": None,
        "ncu_metric_version": metric_availability.get("ncu_metric_version"),
        "manifest_validation": {"status": "not_run"},
        "provenance_audit": {"status": "not_run"},
        "preflight_status": "excluded",
        "typed_reason": None,
        "detail": None,
    }

    if not binary_path.is_file():
        entry["typed_reason"] = "binary_missing"
        entry["detail"] = f"binary not found at {entry['binary_path']}"
        return entry
    entry["binary_sha256"] = _sha256(binary_path)

    if not trace_path.is_file():
        entry["typed_reason"] = "predictor_trace_missing"
        entry["detail"] = f"trace JSON not found at {entry['expected_predictor_trace_path']}"
        return entry
    try:
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        entry["typed_reason"] = "predictor_trace_parse_error"
        entry["detail"] = str(exc)
        return entry
    if not isinstance(trace_payload, dict):
        entry["typed_reason"] = "predictor_trace_parse_error"
        entry["detail"] = "trace JSON root is not an object"
        return entry
    identity_failure = _trace_identity_failure(row=row, trace_payload=trace_payload)
    if identity_failure is not None:
        entry["typed_reason"] = "predictor_trace_identity_mismatch"
        entry["detail"] = identity_failure
        return entry
    entry["source_commit"] = trace_payload.get("source_commit")
    entry["kernel_signature_sha256"] = trace_payload.get(
        "kernel_signature_sha256"
    )

    try:
        _stage_manifest_from_trace(
            trace_path=trace_path,
            manifest_path=manifest_path,
            sku=sku,
            shape_point=shape_point,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        entry["typed_reason"] = "manifest_staging_failure"
        entry["detail"] = str(exc)
        return entry
    entry["staged_manifest_path"] = _rel(manifest_path, repo_root)

    valid, validation_reason, validation_detail = _validate_manifest_against_binary(
        manifest_path=manifest_path, binary_path=binary_path, arch_dir=arch_dir
    )
    entry["manifest_validation"] = {
        "status": "pass" if valid else "fail",
        "typed_reason": validation_reason,
        "detail": validation_detail,
    }
    if not valid:
        entry["typed_reason"] = validation_reason
        entry["detail"] = validation_detail
        return entry

    try:
        audit = audit_runner(
            predictor_trace_json=trace_path,
            manifest=manifest_path,
            arch=arch_dir,
            repo_root=repo_root,
            live_sweep=live_sweep,
        )
    except Exception as exc:  # provenance_audit maps CLI IO failures to exit 2.
        entry["provenance_audit"] = {
            "status": "fail",
            "typed_reason": "provenance_audit_io_failure",
            "detail": str(exc),
            "exit_code": 2,
        }
        entry["typed_reason"] = "provenance_audit_io_failure"
        entry["detail"] = str(exc)
        return entry
    entry["provenance_audit"] = {
        "status": "pass" if audit.exit_code == 0 else "fail",
        "typed_reason": audit.typed_reason,
        "detail": audit.detail,
        "exit_code": audit.exit_code,
        "live_sweep": live_sweep,
    }
    if audit.exit_code != 0:
        entry["typed_reason"] = audit.typed_reason or "provenance_audit_failure"
        entry["detail"] = audit.detail
        return entry

    if not metric_availability.get("confirmed"):
        entry["typed_reason"] = "metric_availability_unconfirmed"
        entry["detail"] = (
            "NCU metric availability artifact is missing or unconfirmed"
        )
        return entry

    entry["preflight_status"] = "live_ready"
    entry["typed_reason"] = None
    entry["detail"] = "trace, manifest, provenance, and metric availability pass"
    return entry


def _floor_preservation(
    *, roster_payload: dict[str, Any], preflight_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    thresholds = roster_payload.get("thresholds", {})
    family_minima = thresholds.get("family_bucket_minima", {})
    live_rows = [r for r in preflight_rows if r["preflight_status"] == "live_ready"]
    primary = [r for r in live_rows if r["role"] == "primary"]
    replacements = [r for r in live_rows if r["role"] == "replacement"]
    primary_slugs = {r["slug"] for r in primary}
    family_slugs: dict[str, set[str]] = collections.defaultdict(set)
    slug_shapes: dict[str, set[str]] = collections.defaultdict(set)
    for row in primary:
        family_slugs[row["family_bucket"]].add(row["slug"])
        slug_shapes[row["slug"]].add(row["shape_point"])
    depth_min = int(thresholds.get("depth_first_min_shape_points", 0))
    depth_first = sorted(
        slug for slug, shapes in slug_shapes.items() if len(shapes) >= depth_min
    )
    checks = {
        "primary_cells": {
            "observed": len(primary),
            "required": thresholds.get("min_primary_cells"),
            "pass": len(primary) >= int(thresholds.get("min_primary_cells", 0)),
        },
        "primary_unique_slugs": {
            "observed": len(primary_slugs),
            "required": thresholds.get("min_primary_unique_slugs"),
            "pass": len(primary_slugs)
            >= int(thresholds.get("min_primary_unique_slugs", 0)),
        },
        "replacement_cells": {
            "observed": len(replacements),
            "required": thresholds.get("min_replacement_cells"),
            "pass": len(replacements)
            >= int(thresholds.get("min_replacement_cells", 0)),
        },
        "depth_first_slugs": {
            "observed": len(depth_first),
            "required": thresholds.get("depth_first_required_slugs"),
            "depth_first_min_shape_points": depth_min,
            "slugs": depth_first,
            "pass": len(depth_first)
            >= int(thresholds.get("depth_first_required_slugs", 0)),
        },
        "family_bucket_primary_unique_slugs": {},
    }
    family_checks = checks["family_bucket_primary_unique_slugs"]
    for family, required in family_minima.items():
        observed = len(family_slugs.get(family, set()))
        family_checks[family] = {
            "observed": observed,
            "required": required,
            "pass": observed >= int(required),
        }
    passes = all(
        check["pass"]
        for key, check in checks.items()
        if key != "family_bucket_primary_unique_slugs"
    ) and all(check["pass"] for check in family_checks.values())
    return {"pass": passes, "checks": checks}


def _failed_floor_dimensions(floor: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    checks = floor.get("checks", {})
    for name, check in checks.items():
        if name == "family_bucket_primary_unique_slugs":
            for family, family_check in check.items():
                if not family_check.get("pass"):
                    failed.append(
                        {
                            "dimension": f"{name}.{family}",
                            "observed": family_check.get("observed"),
                            "required": family_check.get("required"),
                        }
                    )
            continue
        if not check.get("pass"):
            failed.append(
                {
                    "dimension": name,
                    "observed": check.get("observed"),
                    "required": check.get("required"),
                }
            )
    return failed


def _format_failed_floor_dimensions(failed: list[dict[str, Any]]) -> str:
    if not failed:
        return "unknown"
    return "; ".join(
        f"{item['dimension']} observed={item['observed']} "
        f"required={item['required']}"
        for item in failed
    )


def build_preflight(
    *,
    roster_path: Path = DEFAULT_ROSTER,
    out_path: Path = DEFAULT_OUT,
    metric_availability_path: Path = DEFAULT_METRIC_AVAILABILITY,
    repo_root: Path = _REPO_ROOT,
    sku: str = "rtx_3090_ti",
    arch_dir: str = "sm_86",
    audit_runner: AuditRunner = provenance_audit.run_audit,
    live_sweep: bool = False,
) -> dict[str, Any]:
    roster_payload = json.loads(roster_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in roster_payload.get("rows", [])
        if row.get("sm86_compatibility_status") == "compatible"
    ]
    metric_availability = _load_metric_availability(
        metric_availability_path, repo_root
    )
    preflight_rows = [
        _preflight_row(
            row=row,
            repo_root=repo_root,
            sku=sku,
            arch_dir=arch_dir,
            metric_availability=metric_availability,
            audit_runner=audit_runner,
            live_sweep=live_sweep,
        )
        for row in rows
    ]
    status_counts = collections.Counter(
        row["preflight_status"] for row in preflight_rows
    )
    reason_counts = collections.Counter(
        row["typed_reason"] or "none" for row in preflight_rows
    )
    artifact = {
        "schema_version": 1,
        "artifact_kind": "sm86_submission_candidate_preflight",
        "generated_at_utc": _utc_now(),
        "source_commit": _git_head(repo_root),
        "scope": {
            "arch": arch_dir,
            "gpu_sku": sku,
            "roster_path": _rel(roster_path, repo_root),
            "metric_availability_path": _rel(metric_availability_path, repo_root),
            "provenance_audit_live_sweep": live_sweep,
        },
        "metric_availability": metric_availability,
        "summary": {
            "compatible_rows": len(rows),
            "live_ready_rows": status_counts.get("live_ready", 0),
            "excluded_rows": status_counts.get("excluded", 0),
            "status_counts": dict(sorted(status_counts.items())),
            "typed_reason_counts": dict(sorted(reason_counts.items())),
            "floor_preservation_after_preflight": _floor_preservation(
                roster_payload=roster_payload,
                preflight_rows=preflight_rows,
            ),
        },
        "rows": preflight_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--metric-availability", type=Path, default=DEFAULT_METRIC_AVAILABILITY
    )
    parser.add_argument("--sku", default="rtx_3090_ti")
    parser.add_argument("--arch-dir", default="sm_86")
    parser.add_argument(
        "--live-sweep",
        action="store_true",
        help="Require predictor source_commit to match HEAD exactly.",
    )
    parser.add_argument(
        "--allow-floor-miss",
        action="store_true",
        help="Write the artifact even when the live-ready floor gate fails.",
    )
    return parser.parse_args(argv)


def run_preflight_cli(
    argv: list[str] | None = None,
    *,
    repo_root: Path = _REPO_ROOT,
    audit_runner: AuditRunner | None = None,
) -> int:
    args = _parse_args(argv)
    artifact = build_preflight(
        roster_path=args.roster,
        out_path=args.out,
        metric_availability_path=args.metric_availability,
        repo_root=repo_root,
        sku=args.sku,
        arch_dir=args.arch_dir,
        audit_runner=audit_runner or provenance_audit.run_audit,
        live_sweep=args.live_sweep,
    )
    summary = artifact["summary"]
    print(
        "submission_candidate_preflight: "
        f"live_ready={summary['live_ready_rows']} "
        f"excluded={summary['excluded_rows']} "
        f"out={args.out}"
    )
    floor = summary["floor_preservation_after_preflight"]
    if not floor["pass"]:
        failed = _failed_floor_dimensions(floor)
        print(
            "submission_candidate_preflight: "
            "floor_preservation_after_preflight failed: "
            f"{_format_failed_floor_dimensions(failed)}",
            file=sys.stderr,
        )
        if not args.allow_floor_miss:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_preflight_cli(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
