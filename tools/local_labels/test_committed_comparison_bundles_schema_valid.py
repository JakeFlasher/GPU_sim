"""Structural-guard test: every committed comparison_bundle.json under
``data/local_labels/sweeps/`` MUST validate against
``workloads/simbench_bw/schemas/comparison_bundle.schema.json``.

This is the audit-trail invariant for the comparison-bundle wire
contract: when the schema bumps a field from optional to required (or
adds a new required field), every committed bundle that ships with the
schema MUST also carry that field, otherwise the repo holds
schema-invalid data alongside the schema that rejects it.

A prior schema-bump-and-backfill slice added ``top_level_admission_scope``
as a required field but backfilled only the production sweep's bundle,
leaving an older sweep's bundle
(``sweep_20260511T015643Z_2f232e09acea``) schema-invalid. This test
prevents that class of drift by walking every committed bundle directly
via filesystem glob (no opt-in flag).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads" / "simbench_bw" / "schemas"
    / "comparison_bundle.schema.json"
)
_SWEEPS_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"


def _committed_comparison_bundles() -> list[Path]:
    """Walk the committed sweeps tree and return every
    ``comparison_bundle.json`` path. The helper covers BOTH layout
    families that historical and current sweeps use:
      - legacy nested layout:
        ``data/local_labels/sweeps/<sweep_id>/comparison/comparison_bundle.json``
      - current root-level layout (used by the
        ``run_headline_cohort_predict_compare.py`` orchestrator's
        ``--output-dir`` convention):
        ``data/local_labels/sweeps/<sweep_id>/comparison_bundle.json``
    Paths are de-duplicated and sorted for deterministic iteration
    order. A future scanner that adds a third layout family must
    extend the glob list here in lockstep, OR a regression test
    should fail on the missing path."""
    if not _SWEEPS_ROOT.is_dir():
        return []
    combined: set[Path] = set()
    combined.update(_SWEEPS_ROOT.glob("*/comparison/comparison_bundle.json"))
    combined.update(_SWEEPS_ROOT.glob("*/comparison_bundle.json"))
    return sorted(combined)


def test_every_committed_comparison_bundle_schema_validates() -> None:
    """Every committed ``comparison_bundle.json`` MUST schema-validate
    against the current ``comparison_bundle.schema.json``.

    Historical regression case: the older sweep
    ``sweep_20260511T015643Z_2f232e09acea`` was missing
    ``top_level_admission_scope`` after the schema-bump-and-backfill
    slice covered only the production sweep. The schema marks the
    field as REQUIRED, so the old bundle failed validation with
    ``'top_level_admission_scope' is a required property``.

    The test loads each bundle, validates against the schema, and
    accumulates per-bundle errors so the failure message names every
    drifted file (not just the first one).
    """
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bundles = _committed_comparison_bundles()
    assert bundles, (
        f"no committed comparison_bundle.json files found under "
        f"{_SWEEPS_ROOT}; the test cannot verify the schema-validity "
        f"invariant without at least one committed bundle"
    )
    failures: list[str] = []
    for bundle_path in bundles:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        try:
            jsonschema.validate(bundle, schema)
        except jsonschema.ValidationError as exc:
            relpath = bundle_path.relative_to(_REPO_ROOT)
            failures.append(f"{relpath}: {exc.message}")
    assert not failures, (
        f"every committed comparison_bundle.json MUST schema-validate "
        f"against comparison_bundle.schema.json; got "
        f"{len(failures)} failure(s):\n  - "
        + "\n  - ".join(failures)
    )


def test_committed_bundles_helper_includes_root_level_layout(tmp_path, monkeypatch) -> None:
    """Regression: sweeps emitted by
    `tools/local_labels/run_headline_cohort_predict_compare.py` land
    at the root-level path
    `data/local_labels/sweeps/<sweep_id>/comparison_bundle.json`
    (NOT under a nested `comparison/` subdirectory). A prior version
    of `_committed_comparison_bundles()` globbed only the legacy
    nested path and silently skipped the root-level bundle; the
    scanner passed without ever validating the new artifact.

    The original pin target (`sweep_20260517T111017Z_57b09f4`) was moved
    to the gitignored data annex in the 2026-07-06 pivot-baseline sweep
    cleanup, so the glob-coverage invariant is now pinned against a
    synthetic sweeps tree carrying BOTH layout families: a future glob
    narrowing still fails here, without requiring the annexed artifact."""
    import sys

    module = sys.modules[_committed_comparison_bundles.__module__]
    fake_sweeps = tmp_path / "sweeps"
    nested = fake_sweeps / "sweep_nested" / "comparison" / "comparison_bundle.json"
    root_level = fake_sweeps / "sweep_root_level" / "comparison_bundle.json"
    for p in (nested, root_level):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "_SWEEPS_ROOT", fake_sweeps)
    bundles = _committed_comparison_bundles()
    assert root_level in bundles and nested in bundles, (
        "the committed-bundles helper MUST cover BOTH the legacy nested "
        "layout (<sweep_id>/comparison/comparison_bundle.json) and the "
        "root-level layout (<sweep_id>/comparison_bundle.json); observed: "
        + ", ".join(str(b.relative_to(tmp_path)) for b in bundles)
    )
