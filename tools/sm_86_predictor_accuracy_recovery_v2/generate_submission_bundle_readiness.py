"""Generate the sm_86 submission bundle-readiness report.

This is an offline inventory. It does not invoke NCU and it does not
run the predictor walker. The report joins the current preflight rows,
admitted labels, recapture ledger, cohort manifest, historical regen
ledger, and on-disk predictor bundles so the next live-labeling pass has
a closed denominator.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
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
DEFAULT_LABEL_ADMITTED = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
DEFAULT_RECAPTURE_LEDGER = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "sm86_label_recapture_sweep_ledger.json"
)
DEFAULT_COHORT_MANIFEST = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_manifest.json"
)
DEFAULT_REGEN_LEDGER = (
    _REPO_ROOT / "data" / "local_labels" / "sm86_cohort_regen_ledger.json"
)
DEFAULT_RECOVERY_ATTEMPTS = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "submission_bundle_recovery_attempts.json"
)
DEFAULT_SWEEPS_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"
DEFAULT_OUT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "submission_bundle_readiness_report.json"
)

IDENTITY_FIELDS = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)
JOIN_CHECK_FIELDS = (
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)
CANONICAL_FEATURE_KEYS = (
    "raw_prediction",
    "footprint_ratio",
    "long_rd_tail_mass",
    "intra_lane_reuse",
    "global_share",
    "gridDim_density",
)
CLOSED_TYPED_REASONS = (
    "missing_bundle",
    "bundle_failed",
    "cache_model_underconfigured",
    "canonical_pre_calibration_features_absent",
    "binary_sha256_mismatch",
    "trace_or_sidecar_absence",
    "join_key_mismatch",
)
_REASON_PRIORITY = {
    reason: index
    for index, reason in enumerate(
        (
            "cache_model_underconfigured",
            "canonical_pre_calibration_features_absent",
            "bundle_failed",
            "binary_sha256_mismatch",
            "trace_or_sidecar_absence",
            "join_key_mismatch",
            "missing_bundle",
        )
    )
}


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


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _load_json(path)


def _cell_key(arch: str, slug: str, shape_point: str) -> tuple[str, str, str]:
    return (str(arch), str(slug), str(shape_point))


def _cell_key_text(key: tuple[str, str, str]) -> str:
    return f"{key[0]}/{key[1]}/{key[2]}"


def _ledger_cell_key(cell_key: str) -> tuple[str, str, str] | None:
    parts = str(cell_key).split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return ("sm_86", parts[0], parts[1])


def _identity_from_mapping(mapping: dict[str, Any]) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for field in IDENTITY_FIELDS:
        value = mapping.get(field)
        if not isinstance(value, str) or not value:
            return None
        out[field] = value
    return out


def _label_identity(cell: dict[str, Any] | None) -> dict[str, str] | None:
    if cell is None:
        return None
    return _identity_from_mapping(cell)


def _preflight_identity(row: dict[str, Any] | None) -> dict[str, str] | None:
    if row is None:
        return None
    return _identity_from_mapping(row)


def _bundle_identity(bundle: dict[str, Any]) -> dict[str, str] | None:
    mpi = bundle.get("manifest_pairing_index")
    if not isinstance(mpi, dict):
        return None
    return _identity_from_mapping(mpi)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_arch_from_bundle_path(path: Path) -> str | None:
    prefix = path.name.split("__", 1)[0]
    return prefix if prefix.startswith("sm_") else None


def _scan_bundles(
    *, sweeps_root: Path, repo_root: Path
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    if not sweeps_root.is_dir():
        return index
    for path in sorted(sweeps_root.rglob("predictor_bundles/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            index[("unknown", "unknown", "unknown")].append(
                {
                    "path": _rel(path, repo_root),
                    "payload": None,
                    "identity": None,
                    "load_error": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        identity = _bundle_identity(payload)
        if identity is None:
            continue
        mpi = payload.get("manifest_pairing_index") or {}
        arch = mpi.get("arch") or _infer_arch_from_bundle_path(path) or "sm_86"
        key = _cell_key(arch, identity["slug"], identity["shape_point"])
        index[key].append(
            {
                "path": _rel(path, repo_root),
                "payload": payload,
                "identity": identity,
                "sha256": _sha256(path),
            }
        )
    return index


def _feature_block_reason(bundle: dict[str, Any]) -> str | None:
    features = bundle.get("canonical_pre_calibration_features")
    if not isinstance(features, dict):
        return "feature block is absent"
    missing = [key for key in CANONICAL_FEATURE_KEYS if key not in features]
    if missing:
        return "feature block misses keys: " + ",".join(missing)
    for key in CANONICAL_FEATURE_KEYS:
        value = features[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"feature {key} is not numeric"
        if not math.isfinite(float(value)):
            return f"feature {key} is not finite"
    return None


def _normalize_reasons(text: str | None, *, status: str | None = None) -> list[str]:
    raw = f"{status or ''} {text or ''}".lower()
    reasons: list[str] = []
    if not raw.strip():
        return reasons
    if "cache_model_underconfigured" in raw:
        reasons.append("cache_model_underconfigured")
    if "canonical_pre_calibration_features" in raw:
        reasons.append("canonical_pre_calibration_features_absent")
    if (
        "bundle_failed" in raw
        or "all_bundle_candidates_rejected" in raw
        or "all_legacy_candidates_rejected" in raw
        or "walker_raised" in raw
        or "schema_invalid" in raw
    ):
        reasons.append("bundle_failed")
    if "no_legacy_5tuple_match" in raw or "missing_bundle" in raw:
        reasons.append("missing_bundle")
    if (
        "binary_sha256" in raw
        and (
            "mismatch" in raw
            or "quarantined" in raw
            or "drift" in raw
            or "diff" in raw
        )
    ):
        reasons.append("binary_sha256_mismatch")
    if (
        "trace_dir_unresolved" in raw
        or "required_sidecars_missing" in raw
        or "trace_missing" in raw
        or "sidecar" in raw
    ):
        reasons.append("trace_or_sidecar_absence")
    if (
        "label_5tuple_unresolved" in raw
        or "join" in raw
        or "source_commit" in raw
        or "kernel_signature_sha256" in raw
        or "manifest_pairing_index" in raw
        or "legacy_bundle_quarantined" in raw
    ):
        reasons.append("join_key_mismatch")
    if not reasons:
        reasons.append("bundle_failed")
    return _sorted_reasons(reasons)


def _sorted_reasons(reasons: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    unique = {reason for reason in reasons if reason in CLOSED_TYPED_REASONS}
    return sorted(unique, key=lambda reason: _REASON_PRIORITY[reason])


def _add_blocker(
    blockers: list[dict[str, Any]],
    *,
    reason: str,
    detail: str,
    source: str,
    terminal: bool,
) -> None:
    if reason not in CLOSED_TYPED_REASONS:
        reason = "bundle_failed"
    item = {
        "reason": reason,
        "detail": detail,
        "source": source,
        "terminal": terminal,
    }
    if item not in blockers:
        blockers.append(item)


def _trace_dir_from_preflight(row: dict[str, Any], repo_root: Path) -> Path:
    trace_json = repo_root / str(row["expected_predictor_trace_path"])
    return trace_json.with_suffix("")


def _trace_sidecar_state(
    *, row: dict[str, Any] | None, repo_root: Path
) -> dict[str, Any]:
    if row is None:
        return {"checked": False, "all_present": None, "paths": {}}
    trace_dir = _trace_dir_from_preflight(row, repo_root)
    required = {
        "dynamic_trace_pb": trace_dir / "dynamic_trace.pb",
        "sharing_locality": trace_dir
        / "sharing_locality"
        / "sharing_locality.json",
        "hrd_profile": trace_dir / "hrd_profile" / "hrd_profile.json",
        "sector_stream": trace_dir / "sector_stream" / "sector_stream.jsonl",
        "sector_stream_mpi": trace_dir
        / "sector_stream"
        / "manifest_pairing_index.json",
    }
    paths = {
        name: {"path": _rel(path, repo_root), "present": path.is_file()}
        for name, path in required.items()
    }
    return {
        "checked": True,
        "trace_dir": _rel(trace_dir, repo_root),
        "all_present": all(item["present"] for item in paths.values()),
        "paths": paths,
    }


def _bundle_gate_reasons(bundle: dict[str, Any]) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    result_kind = bundle.get("result_kind")
    typed_reason = bundle.get("typed_reason")
    if typed_reason:
        for reason in _normalize_reasons(str(typed_reason)):
            reasons.append({"reason": reason, "detail": str(typed_reason)})
    if result_kind not in (None, "scalar"):
        for reason in _normalize_reasons(str(result_kind), status="bundle_failed"):
            reasons.append({"reason": reason, "detail": f"result_kind={result_kind}"})
    if "predicted_hit_rate" not in bundle and result_kind in (None, "scalar"):
        reasons.append(
            {
                "reason": "bundle_failed",
                "detail": "scalar bundle has no predicted_hit_rate",
            }
        )
    feature_reason = _feature_block_reason(bundle)
    if feature_reason is not None:
        reasons.append(
            {
                "reason": "canonical_pre_calibration_features_absent",
                "detail": feature_reason,
            }
        )
    return reasons


def _identity_mismatches(
    *, expected: dict[str, str], actual: dict[str, str]
) -> list[str]:
    return [
        field
        for field in JOIN_CHECK_FIELDS
        if expected.get(field) != actual.get(field)
    ]


def _evaluate_bundle_locations(
    *,
    key: tuple[str, str, str],
    expected_identity: dict[str, str],
    bundle_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    preflight_live: bool,
    sidecars_present: bool,
) -> dict[str, Any]:
    candidates = list(bundle_index.get(key, []))
    full_matches = [
        item for item in candidates if item.get("identity") == expected_identity
    ]
    stale_candidates = [
        item for item in candidates if item.get("identity") != expected_identity
    ]
    ready: dict[str, Any] | None = None
    rejected: list[dict[str, Any]] = []
    rejected_blockers: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for item in full_matches:
        payload = item.get("payload")
        if not isinstance(payload, dict):
            gate_reasons = [
                {
                    "reason": "bundle_failed",
                    "detail": f"bundle JSON could not be loaded at {item['path']}",
                }
            ]
            rejected.append(
                {
                    "path": item["path"],
                    "typed_reasons": ["bundle_failed"],
                    "details": gate_reasons,
                }
            )
            for entry in gate_reasons:
                rejected_blockers.append(
                    {
                        "reason": entry["reason"],
                        "detail": entry["detail"],
                        "source": "predictor_bundle",
                        "terminal": True,
                    }
                )
            continue
        gate_reasons = _bundle_gate_reasons(payload)
        if not gate_reasons:
            ready = item
            break
        rejected.append(
            {
                "path": item["path"],
                "typed_reasons": _sorted_reasons(
                    [entry["reason"] for entry in gate_reasons]
                ),
                "details": gate_reasons,
            }
        )
        for entry in gate_reasons:
            rejected_blockers.append(
                {
                    "reason": entry["reason"],
                    "detail": f"{item['path']}: {entry['detail']}",
                    "source": "predictor_bundle",
                    "terminal": True,
                }
            )
    if ready is None:
        for entry in rejected_blockers:
            _add_blocker(blockers, **entry)
    if ready is None and not full_matches:
        if not candidates:
            _add_blocker(
                blockers,
                reason="missing_bundle",
                detail="no predictor bundle exists for this cell key",
                source="predictor_bundle",
                terminal=not (preflight_live and sidecars_present),
            )
        else:
            diff_fields: set[str] = set()
            for item in stale_candidates:
                identity = item.get("identity") or {}
                diff_fields.update(
                    _identity_mismatches(
                        expected=expected_identity, actual=identity
                    )
                )
            if "binary_sha256" in diff_fields:
                _add_blocker(
                    blockers,
                    reason="binary_sha256_mismatch",
                    detail=(
                        "same slug/shape bundle exists but binary_sha256 "
                        "does not match the expected identity"
                    ),
                    source="predictor_bundle",
                    terminal=not (preflight_live and sidecars_present),
                )
            if diff_fields - {"binary_sha256"}:
                _add_blocker(
                    blockers,
                    reason="join_key_mismatch",
                    detail=(
                        "same slug/shape bundle exists but join fields drift: "
                        + ",".join(sorted(diff_fields))
                    ),
                    source="predictor_bundle",
                    terminal=not (preflight_live and sidecars_present),
                )
    return {
        "ready": ready is not None,
        "ready_path": ready["path"] if ready else None,
        "ready_bundle_sha256": ready.get("sha256") if ready else None,
        "candidate_paths": [item["path"] for item in candidates],
        "full_match_paths": [item["path"] for item in full_matches],
        "stale_candidate_paths": [item["path"] for item in stale_candidates],
        "rejected_full_matches": rejected,
        "blockers": blockers,
    }


def _index_labels(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            continue
        key = _cell_key(
            cell.get("arch", ""), cell.get("slug", ""), cell.get("shape_point", "")
        )
        if key[0] and key[1] and key[2]:
            out[key] = cell
    return out


def _index_preflight_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload.get("rows", [])):
        if row.get("preflight_status") != "live_ready":
            continue
        item = dict(row)
        item["_preflight_index"] = index
        rows.append(item)
    return rows


def _index_recapture(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            continue
        key = _ledger_cell_key(str(cell.get("cell_key", "")))
        if key is not None:
            out[key] = cell
    return out


def _index_cohort_manifest(
    payload: dict[str, Any]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            continue
        key = _cell_key(
            cell.get("arch", ""), cell.get("slug", ""), cell.get("shape_point", "")
        )
        if key[0] and key[1] and key[2]:
            out[key] = cell
    return out


def _index_regen(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            continue
        key = _cell_key(
            cell.get("arch", ""), cell.get("slug", ""), cell.get("shape_point", "")
        )
        if key[0] and key[1] and key[2]:
            out[key] = cell
    return out


def _index_recovery_attempts(
    payload: dict[str, Any]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for attempt in payload.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        parts = str(attempt.get("cell_key", "")).split("/")
        if len(parts) != 3:
            continue
        key = _cell_key(parts[0], parts[1], parts[2])
        if key[0] and key[1] and key[2]:
            out[key] = attempt
    return out


def _source_rows(
    *,
    live_preflight_rows: list[dict[str, Any]],
    labels: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    preflight_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in live_preflight_rows:
        key = _cell_key(row["arch"], row["slug"], row["shape_point"])
        preflight_by_key[key].append(row)
    for key in sorted(labels):
        if key[0] != "sm_86":
            continue
        rows.append(
            {
                "row_id": "admitted:" + _cell_key_text(key),
                "source_kind": "admitted_label",
                "key": key,
                "label": labels[key],
                "preflight_rows": list(preflight_by_key.get(key, [])),
            }
        )
    for row in live_preflight_rows:
        key = _cell_key(row["arch"], row["slug"], row["shape_point"])
        if key in labels:
            continue
        rows.append(
            {
                "row_id": f"preflight:{row['_preflight_index']}",
                "source_kind": "preflight_candidate",
                "key": key,
                "preflight": row,
                "preflight_rows": [row],
            }
        )
    return rows


def _identity_summary(identity: dict[str, str] | None) -> dict[str, str] | None:
    if identity is None:
        return None
    return {field: identity[field] for field in IDENTITY_FIELDS}


def _classify_instance(
    *,
    source: dict[str, Any],
    labels: dict[tuple[str, str, str], dict[str, Any]],
    recapture: dict[tuple[str, str, str], dict[str, Any]],
    cohort: dict[tuple[str, str, str], dict[str, Any]],
    regen: dict[tuple[str, str, str], dict[str, Any]],
    recovery: dict[tuple[str, str, str], dict[str, Any]],
    bundles: dict[tuple[str, str, str], list[dict[str, Any]]],
    repo_root: Path,
) -> dict[str, Any]:
    key = source["key"]
    source_kind = str(source["source_kind"])
    preflight_rows = list(source.get("preflight_rows") or [])
    preflight_row = preflight_rows[0] if preflight_rows else None
    label_row = source.get("label") or labels.get(key)
    recapture_row = recapture.get(key)
    cohort_row = cohort.get(key)
    regen_row = regen.get(key)
    recovery_row = recovery.get(key)
    preflight_identity = _preflight_identity(preflight_row)
    label_identity = _label_identity(label_row)
    expected_identity = (
        label_identity if source_kind == "admitted_label" else preflight_identity
    )
    blockers: list[dict[str, Any]] = []
    sidecar_state = _trace_sidecar_state(row=preflight_row, repo_root=repo_root)
    sidecars_present = bool(sidecar_state.get("all_present"))
    preflight_live = preflight_row is not None
    preflight_blocks_this_slice = source_kind == "preflight_candidate"

    if preflight_blocks_this_slice and not sidecars_present:
        missing = [
            name
            for name, info in sidecar_state.get("paths", {}).items()
            if not info.get("present")
        ]
        _add_blocker(
            blockers,
            reason="trace_or_sidecar_absence",
            detail="missing trace sidecars: " + ",".join(sorted(missing)),
            source="preflight_trace",
            terminal=True,
        )

    if label_row is None:
        _add_blocker(
            blockers,
            reason="join_key_mismatch",
            detail="no admitted-label row exists for this arch/slug/shape",
            source="label_admitted",
            terminal=True,
        )
    preflight_drift_fields: list[str] = []
    if preflight_identity is not None and label_identity is not None:
        diff_fields = _identity_mismatches(
            expected=preflight_identity, actual=label_identity
        )
        preflight_drift_fields = sorted(diff_fields)
        if source_kind == "preflight_candidate" and "binary_sha256" in diff_fields:
            _add_blocker(
                blockers,
                reason="binary_sha256_mismatch",
                detail=(
                    "preflight binary_sha256 and admitted-label "
                    "binary_sha256 differ"
                ),
                source="label_admitted",
                terminal=False,
            )
        other = sorted(set(diff_fields) - {"binary_sha256"})
        if source_kind == "preflight_candidate" and other:
            _add_blocker(
                blockers,
                reason="join_key_mismatch",
                detail=(
                    "preflight and admitted-label join fields differ: "
                    + ",".join(other)
                ),
                source="label_admitted",
                terminal=False,
            )

    if recapture_row is None and source_kind == "preflight_candidate":
        _add_blocker(
            blockers,
            reason="join_key_mismatch",
            detail="no recapture-ledger row exists for this arch/slug/shape",
            source="recapture_ledger",
            terminal=True,
        )
    elif recapture_row is not None and recapture_row.get("status") != "admitted":
        for reason in _normalize_reasons(
            str(recapture_row.get("typed_failure_reason")),
            status=str(recapture_row.get("status")),
        ):
            _add_blocker(
                blockers,
                reason=reason,
                detail=(
                    f"recapture status={recapture_row.get('status')} "
                    f"reason={recapture_row.get('typed_failure_reason')}"
                ),
                source="recapture_ledger",
                terminal=True,
            )

    if regen_row is not None and regen_row.get("status") == "bundle_failed":
        for reason in _normalize_reasons(
            str(regen_row.get("typed_reason")),
            status=str(regen_row.get("status")),
        ):
            _add_blocker(
                blockers,
                reason=reason,
                detail=(
                    f"historical bundle status=bundle_failed "
                    f"reason={regen_row.get('typed_reason')}"
                ),
                source="regen_ledger",
                terminal=True,
            )

    if recovery_row is not None and recovery_row.get("status") != "admitted":
        for reason in _normalize_reasons(
            str(recovery_row.get("typed_failure_reason")),
            status=str(recovery_row.get("status")),
        ):
            _add_blocker(
                blockers,
                reason=reason,
                detail=(
                    f"recovery status={recovery_row.get('status')} "
                    f"reason={recovery_row.get('typed_failure_reason')}"
                ),
                source="bundle_recovery_attempt",
                terminal=True,
            )

    bundle_state: dict[str, Any]
    if expected_identity is None:
        bundle_state = {
            "ready": False,
            "ready_path": None,
            "ready_bundle_sha256": None,
            "candidate_paths": [],
            "full_match_paths": [],
            "stale_candidate_paths": [],
            "rejected_full_matches": [],
            "blockers": [
                {
                    "reason": "join_key_mismatch",
                    "detail": "expected bundle identity could not be formed",
                    "source": "predictor_bundle",
                    "terminal": True,
                }
            ],
        }
    else:
        bundle_state = _evaluate_bundle_locations(
            key=key,
            expected_identity=expected_identity,
            bundle_index=bundles,
            preflight_live=False,
            sidecars_present=False,
        )
    for blocker in bundle_state["blockers"]:
        _add_blocker(blockers, **blocker)

    terminal_blockers = [blocker for blocker in blockers if blocker["terminal"]]
    typed_reasons = _sorted_reasons(
        [blocker["reason"] for blocker in blockers]
    )
    if bundle_state["ready"] and not blockers:
        classification = "bundle_ready"
    elif terminal_blockers:
        classification = "typed_unavailable"
    elif source_kind == "preflight_candidate" and sidecars_present:
        classification = "recoverable"
    else:
        classification = "typed_unavailable"
    primary_reason = typed_reasons[0] if typed_reasons else None
    return {
        "row_id": source["row_id"],
        "source_kind": source_kind,
        "cell": {
            "arch": key[0],
            "slug": key[1],
            "shape_point": key[2],
        },
        "classification": classification,
        "primary_typed_reason": primary_reason,
        "typed_reasons": typed_reasons,
        "blockers": blockers,
        "preflight": {
            "present": preflight_row is not None,
            "matching_row_count": len(preflight_rows),
            "row_index": (
                preflight_row.get("_preflight_index")
                if preflight_row is not None
                else None
            ),
            "role": preflight_row.get("role") if preflight_row else None,
            "family_bucket": (
                preflight_row.get("family_bucket") if preflight_row else None
            ),
            "binary_sha256": (
                preflight_row.get("binary_sha256") if preflight_row else None
            ),
            "source_commit": (
                preflight_row.get("source_commit") if preflight_row else None
            ),
            "kernel_signature_sha256": (
                preflight_row.get("kernel_signature_sha256")
                if preflight_row
                else None
            ),
            "expected_predictor_trace_path": (
                preflight_row.get("expected_predictor_trace_path")
                if preflight_row
                else None
            ),
            "identity": _identity_summary(preflight_identity),
            "drift_fields_vs_label": preflight_drift_fields,
            "drift_blocks_admitted_readiness": (
                source_kind == "preflight_candidate"
            ),
        },
        "label_admitted": {
            "present": label_row is not None,
            "identity": _identity_summary(label_identity),
        },
        "recapture_ledger": {
            "present": recapture_row is not None,
            "status": recapture_row.get("status") if recapture_row else None,
            "attempted_at_utc": (
                recapture_row.get("attempted_at_utc") if recapture_row else None
            ),
        },
        "cohort_manifest": {
            "present": cohort_row is not None,
            "admission_status": (
                cohort_row.get("admission_status") if cohort_row else None
            ),
            "exclusion_typed_reason": (
                cohort_row.get("exclusion_typed_reason") if cohort_row else None
            ),
        },
        "regen_ledger": {
            "present": regen_row is not None,
            "status": regen_row.get("status") if regen_row else None,
            "typed_reason": regen_row.get("typed_reason") if regen_row else None,
            "predictor_bundle_path": (
                regen_row.get("predictor_bundle_path") if regen_row else None
            ),
        },
        "recovery_attempt": {
            "present": recovery_row is not None,
            "status": recovery_row.get("status") if recovery_row else None,
            "typed_failure_reason": (
                recovery_row.get("typed_failure_reason")
                if recovery_row
                else None
            ),
            "bundle_admission_path": (
                recovery_row.get("bundle_admission_path")
                if recovery_row
                else None
            ),
            "output_path": recovery_row.get("output_path") if recovery_row else None,
        },
        "trace_sidecars": sidecar_state,
        "bundle_readiness": {
            key: value
            for key, value in bundle_state.items()
            if key != "blockers"
        },
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = collections.Counter(row["classification"] for row in rows)
    slice_status_counts: dict[str, dict[str, int]] = {}
    for source_kind in sorted({row["source_kind"] for row in rows}):
        slice_counter = collections.Counter(
            row["classification"]
            for row in rows
            if row["source_kind"] == source_kind
        )
        slice_status_counts[source_kind] = dict(sorted(slice_counter.items()))
    reason_counts: collections.Counter[str] = collections.Counter()
    blocker_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        if not row["typed_reasons"]:
            reason_counts["none"] += 1
        for reason in row["typed_reasons"]:
            reason_counts[reason] += 1
        for blocker in row["blockers"]:
            blocker_counts[blocker["reason"]] += 1
    untyped = [
        {
            "row_id": row["row_id"],
            "cell": row["cell"],
            "classification": row["classification"],
        }
        for row in rows
        if row["classification"] != "bundle_ready" and not row["typed_reasons"]
    ]
    unique_cells = {
        (row["cell"]["arch"], row["cell"]["slug"], row["cell"]["shape_point"])
        for row in rows
    }
    ready_unique = {
        (row["cell"]["arch"], row["cell"]["slug"], row["cell"]["shape_point"])
        for row in rows
        if row["classification"] == "bundle_ready"
    }
    blockers = [
        {
            "row_id": row["row_id"],
            "cell": row["cell"],
            "classification": row["classification"],
            "typed_reasons": row["typed_reasons"],
            "terminal_reasons": _sorted_reasons(
                [
                    blocker["reason"]
                    for blocker in row["blockers"]
                    if blocker["terminal"]
                ]
            ),
        }
        for row in rows
        if row["classification"] != "bundle_ready"
    ]
    return {
        "denominator_rows": len(rows),
        "denominator_unique_cells": len(unique_cells),
        "status_counts": dict(sorted(status_counts.items())),
        "slice_status_counts": slice_status_counts,
        "typed_reason_counts": dict(sorted(reason_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "post_recovery_denominator": {
            "bundle_ready_rows": status_counts.get("bundle_ready", 0),
            "bundle_ready_unique_cells": len(ready_unique),
            "recoverable_rows": status_counts.get("recoverable", 0),
            "typed_unavailable_rows": status_counts.get("typed_unavailable", 0),
            "publication_slice_blocker_rows": len(blockers),
        },
        "no_untyped_publication_slice_failure": not untyped,
        "untyped_publication_slice_failures": untyped,
        "publication_slice_blockers": blockers,
    }


def build_bundle_readiness_report(
    *,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    label_admitted_path: Path = DEFAULT_LABEL_ADMITTED,
    recapture_ledger_path: Path = DEFAULT_RECAPTURE_LEDGER,
    cohort_manifest_path: Path = DEFAULT_COHORT_MANIFEST,
    regen_ledger_path: Path = DEFAULT_REGEN_LEDGER,
    recovery_attempts_path: Path = DEFAULT_RECOVERY_ATTEMPTS,
    sweeps_root: Path = DEFAULT_SWEEPS_ROOT,
    out_path: Path = DEFAULT_OUT,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, Any]:
    preflight_payload = _load_json(preflight_path)
    label_payload = _load_json(label_admitted_path)
    recapture_payload = _load_json(recapture_ledger_path)
    cohort_payload = _load_json(cohort_manifest_path)
    regen_payload = _optional_json(regen_ledger_path)
    recovery_payload = _optional_json(recovery_attempts_path)

    labels = _index_labels(label_payload)
    live_preflight_rows = _index_preflight_rows(preflight_payload)
    recapture = _index_recapture(recapture_payload)
    cohort = _index_cohort_manifest(cohort_payload)
    regen = _index_regen(regen_payload)
    recovery = _index_recovery_attempts(recovery_payload)
    bundle_index = _scan_bundles(sweeps_root=sweeps_root, repo_root=repo_root)
    source_rows = _source_rows(
        live_preflight_rows=live_preflight_rows, labels=labels
    )
    rows = [
        _classify_instance(
            source=source,
            labels=labels,
            recapture=recapture,
            cohort=cohort,
            regen=regen,
            recovery=recovery,
            bundles=bundle_index,
            repo_root=repo_root,
        )
        for source in source_rows
    ]
    artifact = {
        "schema_version": 1,
        "artifact_kind": "sm86_submission_bundle_readiness_report",
        "generated_at_utc": _utc_now(),
        "source_commit": _git_head(repo_root),
        "scope": {
            "arch": "sm_86",
            "preflight_path": _rel(preflight_path, repo_root),
            "label_admitted_path": _rel(label_admitted_path, repo_root),
            "recapture_ledger_path": _rel(recapture_ledger_path, repo_root),
            "cohort_manifest_path": _rel(cohort_manifest_path, repo_root),
            "regen_ledger_path": _rel(regen_ledger_path, repo_root),
            "recovery_attempts_path": _rel(recovery_attempts_path, repo_root),
            "sweeps_root": _rel(sweeps_root, repo_root),
            "live_ncu_invoked": False,
        },
        "closed_typed_reason_enum": list(CLOSED_TYPED_REASONS),
        "input_counts": {
            "preflight_live_ready_rows": len(live_preflight_rows),
            "label_admitted_sm86_rows": sum(
                1 for key in labels if key[0] == "sm_86"
            ),
            "recapture_ledger_rows": len(recapture),
            "cohort_manifest_rows": len(cohort),
            "regen_ledger_rows": len(regen),
            "bundle_recovery_attempt_rows": len(recovery),
            "bundle_files_scanned": sum(len(v) for v in bundle_index.values()),
        },
        "summary": _summary(rows),
        "slices": {
            "admitted_label": [
                row["row_id"]
                for row in rows
                if row["source_kind"] == "admitted_label"
            ],
            "preflight_candidate": [
                row["row_id"]
                for row in rows
                if row["source_kind"] == "preflight_candidate"
            ],
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument(
        "--label-admitted", type=Path, default=DEFAULT_LABEL_ADMITTED
    )
    parser.add_argument(
        "--recapture-ledger", type=Path, default=DEFAULT_RECAPTURE_LEDGER
    )
    parser.add_argument(
        "--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST
    )
    parser.add_argument("--regen-ledger", type=Path, default=DEFAULT_REGEN_LEDGER)
    parser.add_argument(
        "--recovery-attempts", type=Path, default=DEFAULT_RECOVERY_ATTEMPTS
    )
    parser.add_argument("--sweeps-root", type=Path, default=DEFAULT_SWEEPS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = build_bundle_readiness_report(
        preflight_path=args.preflight,
        label_admitted_path=args.label_admitted,
        recapture_ledger_path=args.recapture_ledger,
        cohort_manifest_path=args.cohort_manifest,
        regen_ledger_path=args.regen_ledger,
        recovery_attempts_path=args.recovery_attempts,
        sweeps_root=args.sweeps_root,
        out_path=args.out,
    )
    summary = artifact["summary"]
    print(
        "submission_bundle_readiness: "
        f"bundle_ready={summary['status_counts'].get('bundle_ready', 0)} "
        f"recoverable={summary['status_counts'].get('recoverable', 0)} "
        f"typed_unavailable="
        f"{summary['status_counts'].get('typed_unavailable', 0)} "
        f"out={args.out}"
    )
    return 0 if summary["no_untyped_publication_slice_failure"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
