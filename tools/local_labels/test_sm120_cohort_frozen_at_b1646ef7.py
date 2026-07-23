"""Freeze regression guard for the 30 sm_120 admitted artifacts at
the v5 calibration-head pin commit b1646ef7.

The v5 footprint-aware per-bucket monotonic L2 hit-rate calibration
head was fit at this commit; the 30 sm_120 cells in
``data/local_labels/label_admitted.json`` are the FROZEN reference
against which the cross-arch same-74 closure lane is evaluated. The
regeneration lane on the originating-arch host (RTX 3090 Ti) MUST
NOT mutate any sm_120 derived artifact — neither the bundle path,
nor the bundle's sha256, nor the bundle's ``manifest_pairing_index``
5-tuple identity.

This module captures the b1646ef7 reference values via
``git show`` and asserts each sm_120 cell's predictor bundle is
byte-equivalent at HEAD. It is invoked at TWO points in the
regeneration lane:

1. As part of the host-only pytest sweep BEFORE the regen wrapper
   runs (pre-wrapper baseline).
2. As a dedicated post-wrapper invocation
   ``uv run --with-requirements requirements.txt --with protobuf
   python -m pytest tools/local_labels/test_sm120_cohort_frozen_at_b1646ef7.py -q``
   immediately AFTER the regen wrapper completes.

Drift in any sm_120 cell at either invocation lifts the typed
defect token ``sm_120_cohort_frozen_violation``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FROZEN_COMMIT = "b1646ef7"
_STAGING_MANIFEST = "data/local_labels/v5_cohort_staging_manifest.json"
_LABEL_ADMITTED = "data/local_labels/label_admitted.json"
_DEFECT_TOKEN = "sm_120_cohort_frozen_violation"
_EXPECTED_SM120_COUNT = 30


def _git_show(commit: str, path: str) -> bytes | None:
    """Return file contents at ``<commit>:<path>`` or ``None`` when
    git or the ref is unavailable. The test fails closed on a
    missing reference rather than skipping silently — a freeze
    guard with no reference is worse than no guard at all.
    """
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "show", f"{commit}:{path}"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _sha256_of_blob(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _read_bundle_5tuple(blob: bytes) -> tuple[str, ...] | None:
    """Extract the canonical
    ``manifest_pairing_index.{slug, shape_point, binary_sha256,
    source_commit, kernel_signature_sha256}`` 5-tuple from a
    predictor-bundle blob.

    Returns ``None`` when the blob is not parseable as JSON or
    when any of the five fields is missing or non-string. The
    test treats ``None`` as a drift and surfaces it.
    """
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return None
    mpi = payload.get("manifest_pairing_index")
    if not isinstance(mpi, dict):
        return None
    needed = (
        "slug",
        "shape_point",
        "binary_sha256",
        "source_commit",
        "kernel_signature_sha256",
    )
    out: list[str] = []
    for key in needed:
        val = mpi.get(key)
        if not isinstance(val, str) or not val:
            return None
        out.append(val)
    return tuple(out)


@pytest.fixture(scope="module")
def frozen_sm120_artifacts() -> dict[tuple[str, str], list[dict[str, object]]]:
    """Load the 30 sm_120 frozen artifacts from the b1646ef7 tree.

    For each sm_120 row in the staging manifest at the frozen
    commit, capture the row's ``predictor_bundle.paths`` list and,
    for each path, compute the bundle blob's sha256 + extract its
    ``manifest_pairing_index`` 5-tuple. Returns a dict keyed by
    ``(slug, shape_point)``.
    """
    manifest_blob = _git_show(_FROZEN_COMMIT, _STAGING_MANIFEST)
    if manifest_blob is None:
        pytest.fail(
            f"{_DEFECT_TOKEN}: git ref {_FROZEN_COMMIT!r} or "
            f"{_STAGING_MANIFEST!r} unavailable; cannot establish "
            f"freeze baseline"
        )
    manifest = json.loads(manifest_blob)
    sm120_rows = [r for r in manifest["rows"] if r["arch"] == "sm_120"]
    if len(sm120_rows) != _EXPECTED_SM120_COUNT:
        pytest.fail(
            f"{_DEFECT_TOKEN}: frozen commit {_FROZEN_COMMIT!r} has "
            f"{len(sm120_rows)} sm_120 rows; expected "
            f"{_EXPECTED_SM120_COUNT}"
        )
    frozen: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in sm120_rows:
        key = (row["slug"], row["shape_point"])
        bundle_block = row.get("predictor_bundle", {})
        paths = list(bundle_block.get("paths", []))
        if not paths:
            pytest.fail(
                f"{_DEFECT_TOKEN}: sm_120 row {key!r} at frozen "
                f"commit has no predictor_bundle.paths; freeze "
                f"reference cannot be established"
            )
        entries: list[dict[str, object]] = []
        for path in paths:
            blob = _git_show(_FROZEN_COMMIT, path)
            if blob is None:
                pytest.fail(
                    f"{_DEFECT_TOKEN}: bundle path {path!r} for "
                    f"sm_120 row {key!r} not present in tree at "
                    f"frozen commit {_FROZEN_COMMIT!r}"
                )
            entries.append({
                "path": path,
                "sha256": _sha256_of_blob(blob),
                "five_tuple": _read_bundle_5tuple(blob),
            })
        frozen[key] = entries
    return frozen


def test_sm120_cohort_coverage_matches_label_admitted(
    frozen_sm120_artifacts: dict[tuple[str, str], list[dict[str, object]]],
) -> None:
    """The 30 frozen sm_120 keys cover every sm_120 cell currently
    admitted in ``label_admitted.json`` at HEAD.
    """
    label = json.loads((_REPO_ROOT / _LABEL_ADMITTED).read_text())
    sm120_cells = [c for c in label["cells"] if c["arch"] == "sm_120"]
    assert len(sm120_cells) == _EXPECTED_SM120_COUNT, (
        f"{_DEFECT_TOKEN}: HEAD sm_120 cell count {len(sm120_cells)} != "
        f"frozen baseline {_EXPECTED_SM120_COUNT}"
    )
    label_keys = {(c["slug"], c["shape_point"]) for c in sm120_cells}
    frozen_keys = set(frozen_sm120_artifacts.keys())
    missing = label_keys - frozen_keys
    extra = frozen_keys - label_keys
    assert not missing and not extra, (
        f"{_DEFECT_TOKEN}: sm_120 (slug, shape_point) coverage drift: "
        f"missing_in_frozen={missing!r} extra_in_frozen={extra!r}"
    )


_LABEL_IDENTITY_KEYS: tuple[str, ...] = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)


def _label_5tuple(cell: dict) -> tuple[str, ...]:
    return tuple(str(cell[k]) for k in _LABEL_IDENTITY_KEYS)


def test_sm120_label_admitted_5tuples_unchanged_from_frozen_commit() -> None:
    """Every sm_120 cell's identity 5-tuple at HEAD's
    ``label_admitted.json`` matches the b1646ef7 baseline.

    Coverage drift on ``(slug, shape_point)`` alone is insufficient:
    a cell can keep the same name pair while drifting on
    ``binary_sha256`` (post-rebuild), ``source_commit`` (recapture),
    or ``kernel_signature_sha256`` (kernel-source edit). All five
    identity fields must remain byte-equivalent or the bundle
    binding established at b1646ef7 is no longer load-bearing.
    """
    frozen_blob = _git_show(_FROZEN_COMMIT, _LABEL_ADMITTED)
    if frozen_blob is None:
        pytest.fail(
            f"{_DEFECT_TOKEN}: git ref {_FROZEN_COMMIT!r} or "
            f"{_LABEL_ADMITTED!r} unavailable; cannot verify 5-tuple "
            f"identity drift"
        )
    frozen_label = json.loads(frozen_blob)
    frozen_sm120 = {
        (c["slug"], c["shape_point"]): _label_5tuple(c)
        for c in frozen_label["cells"]
        if c["arch"] == "sm_120"
    }
    if len(frozen_sm120) != _EXPECTED_SM120_COUNT:
        pytest.fail(
            f"{_DEFECT_TOKEN}: frozen label_admitted.json has "
            f"{len(frozen_sm120)} sm_120 cells; expected "
            f"{_EXPECTED_SM120_COUNT}"
        )
    head_label = json.loads((_REPO_ROOT / _LABEL_ADMITTED).read_text())
    head_sm120 = {
        (c["slug"], c["shape_point"]): _label_5tuple(c)
        for c in head_label["cells"]
        if c["arch"] == "sm_120"
    }
    missing = set(frozen_sm120) - set(head_sm120)
    extra = set(head_sm120) - set(frozen_sm120)
    drifted: list[str] = []
    if missing:
        drifted.append(f"missing_at_head={missing!r}")
    if extra:
        drifted.append(f"extra_at_head={extra!r}")
    for key, frozen_id in frozen_sm120.items():
        head_id = head_sm120.get(key)
        if head_id is None:
            continue  # caught above
        if head_id != frozen_id:
            drifted.append(
                f"row {key!r}: 5-tuple frozen={frozen_id!r} "
                f"head={head_id!r}"
            )
    assert not drifted, (
        f"{_DEFECT_TOKEN}: sm_120 label identity 5-tuple drift:\n  "
        + "\n  ".join(drifted)
    )


def test_sm120_bundle_paths_unchanged_from_frozen_commit(
    frozen_sm120_artifacts: dict[tuple[str, str], list[dict[str, object]]],
) -> None:
    """The ``predictor_bundle.paths`` list for each sm_120 row at
    HEAD's staging manifest matches the b1646ef7 baseline exactly,
    in order.
    """
    head_manifest = json.loads((_REPO_ROOT / _STAGING_MANIFEST).read_text())
    head_sm120 = [r for r in head_manifest["rows"] if r["arch"] == "sm_120"]
    head_lookup = {(r["slug"], r["shape_point"]): r for r in head_sm120}
    drifted: list[str] = []
    for key, frozen_entries in frozen_sm120_artifacts.items():
        frozen_paths = [str(e["path"]) for e in frozen_entries]
        head_row = head_lookup.get(key)
        if head_row is None:
            drifted.append(f"row {key!r} absent at HEAD")
            continue
        head_paths = list(head_row.get("predictor_bundle", {}).get("paths", []))
        if head_paths != frozen_paths:
            drifted.append(
                f"row {key!r}: frozen_paths={frozen_paths!r} != "
                f"head_paths={head_paths!r}"
            )
    assert not drifted, (
        f"{_DEFECT_TOKEN}: predictor_bundle.paths drift:\n  "
        + "\n  ".join(drifted)
    )


def test_sm120_bundle_sha256_unchanged_from_frozen_commit(
    frozen_sm120_artifacts: dict[tuple[str, str], list[dict[str, object]]],
) -> None:
    """Each frozen bundle file's sha256 at HEAD matches the
    b1646ef7 baseline.
    """
    drifted: list[str] = []
    for key, frozen_entries in frozen_sm120_artifacts.items():
        for entry in frozen_entries:
            on_disk = _REPO_ROOT / str(entry["path"])
            if not on_disk.is_file():
                drifted.append(
                    f"row {key!r}: frozen bundle {entry['path']!r} "
                    f"missing on disk at HEAD"
                )
                continue
            head_sha = _sha256_of_blob(on_disk.read_bytes())
            if head_sha != entry["sha256"]:
                drifted.append(
                    f"row {key!r}: bundle {entry['path']!r} sha drift "
                    f"frozen={entry['sha256']!r} head={head_sha!r}"
                )
    assert not drifted, (
        f"{_DEFECT_TOKEN}: predictor bundle sha256 drift:\n  "
        + "\n  ".join(drifted)
    )


def test_sm120_bundle_mpi_5tuple_unchanged_from_frozen_commit(
    frozen_sm120_artifacts: dict[tuple[str, str], list[dict[str, object]]],
) -> None:
    """Each frozen bundle's ``manifest_pairing_index`` 5-tuple at
    HEAD matches the b1646ef7 baseline. Identity drift is the
    sharpest signal of a sm_120 frozen-cohort violation.
    """
    drifted: list[str] = []
    for key, frozen_entries in frozen_sm120_artifacts.items():
        for entry in frozen_entries:
            on_disk = _REPO_ROOT / str(entry["path"])
            if not on_disk.is_file():
                continue  # path drift already surfaced upstream
            head_5tuple = _read_bundle_5tuple(on_disk.read_bytes())
            if head_5tuple != entry["five_tuple"]:
                drifted.append(
                    f"row {key!r}: bundle {entry['path']!r} 5-tuple "
                    f"drift frozen={entry['five_tuple']!r} "
                    f"head={head_5tuple!r}"
                )
    assert not drifted, (
        f"{_DEFECT_TOKEN}: predictor bundle manifest_pairing_index "
        f"5-tuple drift:\n  " + "\n  ".join(drifted)
    )
