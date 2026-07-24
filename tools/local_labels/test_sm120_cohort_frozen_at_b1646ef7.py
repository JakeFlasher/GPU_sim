"""Freeze guards for the retained 30-cell sm_120 cohort projection.

Default tests compare checked-in cohort bytes with a direct-path retained
projection.  The exact temporal claim against Git commit ``b1646ef7`` is a
campaign-evidence owner test and fails closed when that history is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FROZEN_COMMIT = "b1646ef7"
_STAGING_MANIFEST = "data/local_labels/v5_cohort_staging_manifest.json"
_LABEL_ADMITTED = "data/local_labels/label_admitted.json"
_RETAINED_PROJECTION = (
    "tools/local_labels/authorities/"
    "sm120_cohort_frozen_at_b1646ef7.v1.json"
)
_DEFECT_TOKEN = "sm_120_cohort_frozen_violation"
_EXPECTED_SM120_COUNT = 30
_EXPECTED_BUNDLE_COUNT = 35
_IDENTITY_KEYS = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)


def _git_show(commit: str, path: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "show", f"{commit}:{path}"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _canonical_json_bytes(document: Any) -> bytes:
    return (
        json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def _safe_relative_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise AssertionError(
            f"{_DEFECT_TOKEN}: unsafe retained predictor-bundle path: {value!r}"
        )
    return value


def _identity_record(payload: Any, *, context: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise AssertionError(f"{_DEFECT_TOKEN}: {context} is not an object")
    identity: dict[str, str] = {}
    for key in _IDENTITY_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise AssertionError(
                f"{_DEFECT_TOKEN}: {context} has invalid {key}: {value!r}"
            )
        identity[key] = value
    return identity


def _json_blob(read_blob: Callable[[str], bytes], path: str) -> dict[str, Any]:
    try:
        document = json.loads(read_blob(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(
            f"{_DEFECT_TOKEN}: cannot load retained JSON {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise AssertionError(f"{_DEFECT_TOKEN}: retained JSON root is not an object: {path}")
    return document


def _build_projection(read_blob: Callable[[str], bytes]) -> dict[str, Any]:
    manifest = _json_blob(read_blob, _STAGING_MANIFEST)
    label = _json_blob(read_blob, _LABEL_ADMITTED)
    manifest_rows = [
        row for row in manifest.get("rows", [])
        if isinstance(row, dict) and row.get("arch") == "sm_120"
    ]
    label_cells = [
        cell for cell in label.get("cells", [])
        if isinstance(cell, dict) and cell.get("arch") == "sm_120"
    ]
    if len(manifest_rows) != _EXPECTED_SM120_COUNT:
        raise AssertionError(
            f"{_DEFECT_TOKEN}: sm_120 staging row count "
            f"{len(manifest_rows)} != {_EXPECTED_SM120_COUNT}"
        )
    if len(label_cells) != _EXPECTED_SM120_COUNT:
        raise AssertionError(
            f"{_DEFECT_TOKEN}: sm_120 label count "
            f"{len(label_cells)} != {_EXPECTED_SM120_COUNT}"
        )

    labels_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for cell in label_cells:
        identity = _identity_record(cell, context="label-admitted cell")
        key = (identity["slug"], identity["shape_point"])
        if key in labels_by_key:
            raise AssertionError(f"{_DEFECT_TOKEN}: duplicate label identity: {key!r}")
        labels_by_key[key] = identity

    projected_rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str]] = set()
    bundle_count = 0
    for row in manifest_rows:
        slug = row.get("slug")
        shape_point = row.get("shape_point")
        if not isinstance(slug, str) or not isinstance(shape_point, str):
            raise AssertionError(f"{_DEFECT_TOKEN}: invalid staging identity")
        key = (slug, shape_point)
        if key in seen_rows:
            raise AssertionError(f"{_DEFECT_TOKEN}: duplicate staging identity: {key!r}")
        seen_rows.add(key)
        label_identity = labels_by_key.get(key)
        if label_identity is None:
            raise AssertionError(
                f"{_DEFECT_TOKEN}: staging identity is absent from labels: {key!r}"
            )
        paths = (row.get("predictor_bundle") or {}).get("paths")
        if not isinstance(paths, list) or not paths:
            raise AssertionError(
                f"{_DEFECT_TOKEN}: staging identity has no bundle paths: {key!r}"
            )
        projected_bundles: list[dict[str, Any]] = []
        for raw_path in paths:
            path = _safe_relative_path(raw_path)
            blob = read_blob(path)
            bundle = _json_blob(lambda _ignored, payload=blob: payload, path)
            pairing = _identity_record(
                bundle.get("manifest_pairing_index"),
                context=f"bundle manifest_pairing_index {path}",
            )
            if pairing != label_identity:
                raise AssertionError(
                    f"{_DEFECT_TOKEN}: bundle/label identity drift at {path}"
                )
            projected_bundles.append({
                "path": path,
                "sha256": _sha256(blob),
                "manifest_pairing_index": pairing,
            })
            bundle_count += 1
        projected_rows.append({
            "slug": slug,
            "shape_point": shape_point,
            "label_identity": label_identity,
            "predictor_bundles": projected_bundles,
        })

    if seen_rows != set(labels_by_key):
        raise AssertionError(
            f"{_DEFECT_TOKEN}: staging/label sm_120 identity-set drift"
        )
    if bundle_count != _EXPECTED_BUNDLE_COUNT:
        raise AssertionError(
            f"{_DEFECT_TOKEN}: predictor-bundle count "
            f"{bundle_count} != {_EXPECTED_BUNDLE_COUNT}"
        )
    projected_rows.sort(key=lambda row: (row["slug"], row["shape_point"]))
    return {
        "schema_version": 1,
        "projection_id": "sm120_cohort_frozen_at_b1646ef7_v1",
        "historical_source_commit": _FROZEN_COMMIT,
        "row_count": _EXPECTED_SM120_COUNT,
        "bundle_count": _EXPECTED_BUNDLE_COUNT,
        "rows": projected_rows,
    }


def _head_blob(path: str) -> bytes:
    return (_REPO_ROOT / _safe_relative_path(path)).read_bytes()


def _retained_projection() -> dict[str, Any]:
    path = _REPO_ROOT / _RETAINED_PROJECTION
    raw = path.read_bytes()
    projection = json.loads(raw)
    assert raw == _canonical_json_bytes(projection), (
        f"{_DEFECT_TOKEN}: retained projection is not canonical JSON"
    )
    assert isinstance(projection, dict)
    assert set(projection) == {
        "schema_version",
        "projection_id",
        "historical_source_commit",
        "row_count",
        "bundle_count",
        "rows",
    }
    assert projection["schema_version"] == 1
    assert projection["projection_id"] == "sm120_cohort_frozen_at_b1646ef7_v1"
    assert projection["historical_source_commit"] == _FROZEN_COMMIT
    assert projection["row_count"] == _EXPECTED_SM120_COUNT
    assert projection["bundle_count"] == _EXPECTED_BUNDLE_COUNT
    return projection


@pytest.fixture(scope="module")
def retained_projection() -> dict[str, Any]:
    return _retained_projection()


@pytest.fixture(scope="module")
def head_projection() -> dict[str, Any]:
    return _build_projection(_head_blob)


def _rows_by_key(projection: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = projection["rows"]
    indexed = {(row["slug"], row["shape_point"]): row for row in rows}
    assert len(indexed) == len(rows) == _EXPECTED_SM120_COUNT
    return indexed


def test_retained_projection_is_exact_direct_path_projection(
    retained_projection: dict[str, Any], head_projection: dict[str, Any],
) -> None:
    assert retained_projection == head_projection


def test_sm120_cohort_coverage_matches_label_admitted(
    retained_projection: dict[str, Any], head_projection: dict[str, Any],
) -> None:
    assert set(_rows_by_key(retained_projection)) == set(_rows_by_key(head_projection))


def test_sm120_label_admitted_5tuples_match_retained_projection(
    retained_projection: dict[str, Any], head_projection: dict[str, Any],
) -> None:
    retained = _rows_by_key(retained_projection)
    head = _rows_by_key(head_projection)
    assert {
        key: row["label_identity"] for key, row in retained.items()
    } == {
        key: row["label_identity"] for key, row in head.items()
    }


def test_sm120_bundle_paths_match_retained_projection(
    retained_projection: dict[str, Any], head_projection: dict[str, Any],
) -> None:
    retained = _rows_by_key(retained_projection)
    head = _rows_by_key(head_projection)
    assert {
        key: [bundle["path"] for bundle in row["predictor_bundles"]]
        for key, row in retained.items()
    } == {
        key: [bundle["path"] for bundle in row["predictor_bundles"]]
        for key, row in head.items()
    }


def test_sm120_bundle_sha256s_match_retained_projection(
    retained_projection: dict[str, Any], head_projection: dict[str, Any],
) -> None:
    retained = _rows_by_key(retained_projection)
    head = _rows_by_key(head_projection)
    assert {
        key: [bundle["sha256"] for bundle in row["predictor_bundles"]]
        for key, row in retained.items()
    } == {
        key: [bundle["sha256"] for bundle in row["predictor_bundles"]]
        for key, row in head.items()
    }


def test_sm120_bundle_mpi_5tuples_match_retained_projection(
    retained_projection: dict[str, Any], head_projection: dict[str, Any],
) -> None:
    retained = _rows_by_key(retained_projection)
    head = _rows_by_key(head_projection)
    assert {
        key: [
            bundle["manifest_pairing_index"]
            for bundle in row["predictor_bundles"]
        ]
        for key, row in retained.items()
    } == {
        key: [
            bundle["manifest_pairing_index"]
            for bundle in row["predictor_bundles"]
        ]
        for key, row in head.items()
    }


@pytest.mark.campaign_evidence
def test_retained_sm120_projection_exactly_rederives_from_b1646ef7() -> None:
    def historical_blob(path: str) -> bytes:
        blob = _git_show(_FROZEN_COMMIT, path)
        if blob is None:
            raise AssertionError(
                f"{_DEFECT_TOKEN}: git ref {_FROZEN_COMMIT!r} or "
                f"path {path!r} is unavailable"
            )
        return blob

    assert _build_projection(historical_blob) == _retained_projection(), (
        f"{_DEFECT_TOKEN}: retained projection differs from {_FROZEN_COMMIT}"
    )
