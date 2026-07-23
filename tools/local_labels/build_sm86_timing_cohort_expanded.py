#!/usr/bin/env python3
"""Derive the EXPANDED sm_86 timing cohort from the committed label
store — the slug-disjoint-scoreable cohort surface.

Admitted rows: every sm_86 admitted-store cell that carries BOTH
timing labels (``gpu__time_duration.sum`` + ``gpc__cycles_elapsed.max``,
each with its captured dispersion / n_repeats / repeatability_rule
admission statistics, RE-CHECKED here against the registry rule
thresholds) at a binary pin that VERIFIES against the on-disk binary
at generation time (the sha is recomputed and recorded as evidence;
the pin itself is MIRRORED from the store row and never hand-edited
— the validator rejects any cohort row whose pin disagrees with the
store, which is the hand-edit recurrence guard).

Typed rows: the legacy long-duration cohort cells whose pinned
binaries no longer exist on disk land as
``timing_binary_sha256_mismatch`` rows carrying the expected and
on-disk shas — cells that cannot bind stay typed, and the validator
rejects any attempt to flip one to admitted (a typed row can never
enter the distinct-slug count).

The distinct-admitted-slug floor (>= 10) is a HARD generation-time
gate: below it the producer refuses to emit (slug-disjoint scoring
would be structurally impossible, which is the very thing this
cohort exists to enable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_LABELS = REPO_ROOT / "tools" / "local_labels"
for p in (str(REPO_ROOT), str(_LOCAL_LABELS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from capture_count_labels import (  # noqa: E402
    _binary_path_for,
    _sha256,
    assert_quarantine_path,
)

DEFAULT_LABEL_ADMITTED = (
    REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
DEFAULT_LEGACY_COHORT = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort"
    / "ac4_long_duration_cohort_sm_86.json"
)
DEFAULT_OUT = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort"
    / "sm86_timing_cohort_expanded.json"
)
DEFAULT_SCHEMA = (
    _LOCAL_LABELS / "sm86_timing_cohort_expanded.schema.json"
)

TIMING_METRICS = (
    "gpu__time_duration.sum",
    "gpc__cycles_elapsed.max",
)
MIN_DISTINCT_SLUGS = 10
_IDENTITY_FIELDS = (
    "arch", "binary_sha256", "source_commit",
    "kernel_signature_sha256", "ncu_metric_version",
)


class TimingCohortFloorError(Exception):
    """Fewer than the required distinct admitted slugs —
    slug-disjoint scoring would be structurally impossible."""


class TimingCohortValidationError(Exception):
    """A cohort row disagrees with the committed store (the
    hand-edit recurrence guard) or a typed row claims admission."""


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _rule_thresholds() -> dict[str, float]:
    from metric_registry import load_registry

    registry = load_registry()
    out = {}
    for metric in TIMING_METRICS:
        entry = registry.get_by_ncu_metric(metric)
        out[metric] = float(registry.threshold_for(entry))
    return out


def build_cohort(
    *,
    label_admitted_path: Path,
    legacy_cohort_path: Path,
) -> dict:
    store = json.loads(
        label_admitted_path.read_text(encoding="utf-8"))
    thresholds = _rule_thresholds()

    admitted_rows: list[dict] = []
    typed_rows: list[dict] = []
    for cell in store.get("cells", []):
        if cell.get("arch") != "sm_86":
            continue
        utm = {
            e["ncu_metric"]: e
            for e in cell.get("unit_tagged_metrics") or []
        }
        timing = {m: utm.get(m) for m in TIMING_METRICS}
        if not all(timing.values()):
            continue  # not a timing-labeled cell
        slug = cell["slug"]
        shape_point = cell["shape_point"]
        identity = {f: cell.get(f) for f in _IDENTITY_FIELDS}

        # Admission-stat re-check against the registry thresholds:
        # the stored dispersion must satisfy its own rule (a
        # violating entry is typed, never silently admitted).
        stat_violations = []
        for metric, entry in timing.items():
            dispersion = entry.get("dispersion")
            n_repeats = entry.get("n_repeats")
            if dispersion is None or n_repeats is None:
                stat_violations.append(
                    f"{metric}: missing admission statistics")
                continue
            if float(dispersion) > thresholds[metric]:
                stat_violations.append(
                    f"{metric}: dispersion {dispersion} exceeds "
                    f"rule threshold {thresholds[metric]}")
        if stat_violations:
            typed_rows.append({
                "slug": slug,
                "shape_point": shape_point,
                **identity,
                "status": "typed",
                "typed_reason": "timing_admission_stats_violated",
                "evidence": {"violations": stat_violations},
            })
            continue

        binary = _binary_path_for(slug, cell.get("binary_sha256"))
        on_disk_sha = _sha256(binary) if binary is not None else None
        if on_disk_sha != cell.get("binary_sha256"):
            typed_rows.append({
                "slug": slug,
                "shape_point": shape_point,
                **identity,
                "status": "typed",
                "typed_reason": "timing_binary_sha256_mismatch",
                "evidence": {
                    "expected_binary_sha256":
                        cell.get("binary_sha256"),
                    "on_disk_binary_sha256": on_disk_sha,
                    "binary_path": (
                        str(binary.relative_to(REPO_ROOT))
                        if binary is not None else None
                    ),
                },
            })
            continue

        admitted_rows.append({
            "slug": slug,
            "shape_point": shape_point,
            **identity,
            "status": "admitted",
            "timing_labels": {
                metric: {
                    "value": entry["value"],
                    "unit": entry["unit"],
                    "dispersion": entry["dispersion"],
                    "n_repeats": entry["n_repeats"],
                    "repeatability_rule":
                        entry["repeatability_rule"],
                    "rule_threshold": thresholds[metric],
                }
                for metric, entry in timing.items()
            },
            "binary_pin_verification": {
                "binary_path": str(binary.relative_to(REPO_ROOT)),
                "on_disk_sha256_at_generation": on_disk_sha,
                "matches_store_pin": True,
            },
        })

    # The legacy long-duration cohort: every cell that cannot bind
    # stays typed with sha evidence (never silently dropped, never
    # admitted without a matching binary).
    legacy = json.loads(
        legacy_cohort_path.read_text(encoding="utf-8"))
    for cell in legacy.get("cells", []):
        slug = cell["slug"]
        shape_point = cell["shape_point"]
        binary = _binary_path_for(slug, cell.get("binary_sha256"))
        on_disk_sha = _sha256(binary) if binary is not None else None
        if on_disk_sha == cell.get("binary_sha256"):
            # A legacy cell that binds would be re-admittable via
            # the standing recapture lane; record it typed-pending
            # rather than silently admitting label values this
            # producer has not re-verified through the store.
            reason = "timing_legacy_cell_binds_pending_recapture"
        else:
            reason = "timing_binary_sha256_mismatch"
        typed_rows.append({
            "slug": slug,
            "shape_point": shape_point,
            "arch": cell.get("arch"),
            "binary_sha256": cell.get("binary_sha256"),
            "source_commit": cell.get("source_commit"),
            "kernel_signature_sha256":
                cell.get("kernel_signature_sha256"),
            "ncu_metric_version": None,
            "status": "typed",
            "typed_reason": reason,
            "evidence": {
                "legacy_cohort": str(
                    legacy_cohort_path.relative_to(REPO_ROOT)),
                "expected_binary_sha256":
                    cell.get("binary_sha256"),
                "on_disk_binary_sha256": on_disk_sha,
            },
        })

    distinct_slugs = sorted(
        {row["slug"] for row in admitted_rows})
    if len(distinct_slugs) < MIN_DISTINCT_SLUGS:
        raise TimingCohortFloorError(
            f"{len(distinct_slugs)} distinct admitted slugs < "
            f"{MIN_DISTINCT_SLUGS}; slug-disjoint scoring is "
            "structurally impossible — refusing to emit"
        )

    return {
        "schema_version": 1,
        "artifact_kind": "sm86_timing_cohort_expanded",
        "arch": "sm_86",
        "generated_at_commit": _git_head(),
        "timing_metrics": list(TIMING_METRICS),
        "rule_thresholds": thresholds,
        "min_distinct_slugs": MIN_DISTINCT_SLUGS,
        "n_admitted": len(admitted_rows),
        "n_typed": len(typed_rows),
        "distinct_admitted_slugs": distinct_slugs,
        "admitted": sorted(
            admitted_rows,
            key=lambda r: (r["slug"], r["shape_point"])),
        "typed": sorted(
            typed_rows,
            key=lambda r: (r["slug"], r["shape_point"],
                           r["typed_reason"])),
    }


def validate_cohort(
    artifact: dict, *, label_admitted_path: Path,
) -> None:
    """The recurrence guard: every admitted row's identity tuple
    (INCLUDING binary_sha256) must equal the committed store row's
    — a hand-edited pin fails closed; a typed row claiming
    admission or entering the slug count fails closed."""
    store = json.loads(
        label_admitted_path.read_text(encoding="utf-8"))
    store_by_key: dict[tuple, dict] = {}
    for cell in store.get("cells", []):
        store_by_key[(
            cell.get("slug"), cell.get("shape_point"),
            cell.get("kernel_signature_sha256"))] = cell
    admitted_slugs = set()
    for row in artifact.get("admitted", []):
        if row.get("status") != "admitted":
            raise TimingCohortValidationError(
                f"non-admitted row in the admitted section: "
                f"{row.get('slug')}/{row.get('shape_point')}"
            )
        key = (row["slug"], row["shape_point"],
               row["kernel_signature_sha256"])
        cell = store_by_key.get(key)
        if cell is None:
            raise TimingCohortValidationError(
                f"admitted cohort row {key[0]}/{key[1]} has no "
                "matching committed store row"
            )
        for field in _IDENTITY_FIELDS:
            if row.get(field) != cell.get(field):
                raise TimingCohortValidationError(
                    f"admitted cohort row {key[0]}/{key[1]} field "
                    f"{field!r} disagrees with the committed store "
                    f"({row.get(field)!r} != {cell.get(field)!r}); "
                    "pins are mirrored, never hand-edited"
                )
        utm = {
            e["ncu_metric"]: e
            for e in cell.get("unit_tagged_metrics") or []
        }
        for metric, mirrored in row["timing_labels"].items():
            entry = utm.get(metric)
            if entry is None or entry["value"] != mirrored["value"]:
                raise TimingCohortValidationError(
                    f"admitted cohort row {key[0]}/{key[1]} timing "
                    f"value for {metric} disagrees with the store"
                )
        admitted_slugs.add(row["slug"])
    for row in artifact.get("typed", []):
        if row.get("status") == "admitted" or row.get(
                "typed_reason") is None:
            raise TimingCohortValidationError(
                f"typed section row {row.get('slug')}/"
                f"{row.get('shape_point')} claims admission"
            )
        if (row["slug"] in admitted_slugs
                and row["typed_reason"]
                == "timing_binary_sha256_mismatch"
                and row["slug"]
                in artifact["distinct_admitted_slugs"]):
            # A slug may legitimately have both an admitted cell
            # and a typed legacy cell; the typed one must simply
            # never be COUNTED — verified below.
            pass
    if sorted(admitted_slugs) != artifact.get(
            "distinct_admitted_slugs"):
        raise TimingCohortValidationError(
            "distinct_admitted_slugs does not equal the admitted "
            "rows' slug set"
        )
    if len(admitted_slugs) < artifact.get(
            "min_distinct_slugs", MIN_DISTINCT_SLUGS):
        raise TimingCohortValidationError(
            "admitted slug count below the declared floor"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-admitted", type=Path,
        default=DEFAULT_LABEL_ADMITTED)
    parser.add_argument(
        "--legacy-cohort", type=Path, default=DEFAULT_LEGACY_COHORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--schema-path", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    assert_quarantine_path(args.out)
    artifact = build_cohort(
        label_admitted_path=args.label_admitted,
        legacy_cohort_path=args.legacy_cohort,
    )
    validate_cohort(
        artifact, label_admitted_path=args.label_admitted)
    import jsonschema
    schema = json.loads(
        args.schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"build_sm86_timing_cohort_expanded: wrote {args.out} "
        f"({artifact['n_admitted']} admitted across "
        f"{len(artifact['distinct_admitted_slugs'])} slugs; "
        f"{artifact['n_typed']} typed)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
