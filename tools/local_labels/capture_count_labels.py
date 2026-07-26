"""Capture deterministic replay-count labels and land them on admitted cells.

Captures every registry replay-count unit-tagged metric that the committed
metric-availability probe confirmed on this device (the global ld/st/atom/red
request counters, the global ld/st sector counters, ``l1tex__t_requests.sum``,
``smsp__inst_executed.sum``, plus the probe-resolved local/texture/surface
request counters) for the admitted sm_86 cells via the local NCU quarantine,
gates them through the SAME per-unit repeatability machinery the label
pipeline uses (``repeatability_aggregator._aggregate_unit_tagged_metrics``
with the registry ``relative_tolerance`` rule) AND a stricter integer-
stability admission on top, then SPLICES the surviving entries into each
matching committed admitted cell's ``unit_tagged_metrics`` IN PLACE —
preserving every other field (L2/L1 hit rates, join keys, ``source_commit``,
existing timing entries) byte-for-byte so the frozen L2 v5 head and its
byte-identity regression guard are not perturbed.

Integer-stability contract (the count-label admission rule):
  - every repeat's value must be integral; a consistently non-integral
    value is typed ``count_value_not_integral`` (a count metric returning
    fractions is misdeclared, not noisy);
  - all repeats must agree EXACTLY (max == min); a counter with
    disagreeing repeats is typed ``label_count_unstable`` with the raw
    repeat values recorded — it is NEVER averaged into a label;
  - the admitted entry's ``dispersion`` is the registry-rule relative
    spread, which is 0.0 by construction for an admitted counter, and the
    recorded per-counter stability statistic is the integer spread
    (max - min == 0) provable from the exclusion records' absence.
  The rule has NO tunable parameters, so no calibration/validation/holdout
  label value can influence any configuration choice (the sealed-holdout
  governance trail records this).

CLAUDE.md compliance (binding):
  - Rule 1 (NCU quarantine + local-label exception): NCU invoked ONLY here
    under ``tools/local_labels/``; every output lands under
    ``data/local_labels/`` (enforced fail-closed by ``main`` via
    ``QuarantinePathError``). ``--metrics`` capture to CSV stdout — no
    ``.ncu-rep`` artifact is written.
  - Rule 2 (no driver-root): ``ncu --metrics`` does not lock clocks; no
    persistence / power / MIG / compute-mode mutation.
  - Rule 3 (no privileged CUPTI): NCU command-line metric capture only.
  - sm_86-only: captures only arch==sm_86 admitted cells, and binds a count
    label ONLY when the on-disk binary's sha256 MATCHES the cell's committed
    ``binary_sha256``; a rebuilt/mismatched binary is a typed exclusion,
    never a silent cross-binary bind.

CLI:
  python tools/local_labels/capture_count_labels.py \
    [--label-admitted data/local_labels/label_admitted.json] \
    [--ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu] \
    [--repeats 5] [--arch sm_86] \
    [--exclusions-out data/local_labels/count_label_exclusions_sm_86.json] \
    [--governance-out data/local_labels/count_label_capture_governance_sm_86.json] \
    [--dry-run]

  Exit 3 (no write) when ncu is not on the host, so a non-3090-Ti host never
  rewrites the committed labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_LOCAL_LABELS = REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS))

from corpus_cell_identity import cell_variant_id  # noqa: E402
from orchestrate_sm86_label_recapture import _app_args_for_cell  # noqa: E402

DEFAULT_LABEL_ADMITTED = "data/local_labels/label_admitted.json"
DEFAULT_EXCLUSIONS_OUT = "data/local_labels/count_label_exclusions_sm_86.json"
DEFAULT_GOVERNANCE_OUT = (
    "data/local_labels/count_label_capture_governance_sm_86.json"
)
DEFAULT_AUDIT_OUT = "data/local_labels/count_label_capture_audit_sm_86.json"
DEFAULT_READMISSION_MANIFEST = (
    "data/local_labels/stale_binary_readmission_manifest_sm_86.json"
)


def _capture_unstable_record(slug: str, shape_point: str) -> dict | None:
    """Return the committed re-admission manifest's capture-unstable
    record for ``slug@shape_point`` (the multi-campaign repeatability
    evidence), or None when the cell is not documented capture-unstable.
    Used to emit self-contained terminal exclusion rows: a consumer of
    the exclusions artifact must not need a side-channel join to learn
    that a cell's count labels are terminally unavailable WITH evidence.
    """
    path = REPO_ROOT / DEFAULT_READMISSION_MANIFEST
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("capture_unstable_cells", {}).get(
        f"{slug}@{shape_point}"
    )
DEFAULT_NCU = "/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu"
DEFAULT_REPEATS = 5
_ARCH = "sm_86"
_QUARANTINE_ROOT = "data/local_labels"
_COHORT_SPLIT = "data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json"
_TOUCH_LEDGER = "data/local_labels/holdout_touch_ledger_sm_86.json"

_BINARY_DIRS = (
    "workloads/simbench_bw/l1",
    "workloads/simbench_bw/l2",
    "workloads/portable_lcd",
)


class QuarantinePathError(ValueError):
    """An output path escapes the data/local_labels quarantine. Count
    labels are NCU-derived and may exist ONLY under the Rule-1 quarantine
    path."""

    typed_reason = "count_label_output_outside_quarantine"


def assert_quarantine_path(path: Path) -> None:
    resolved = path.resolve()
    quarantine = (REPO_ROOT / _QUARANTINE_ROOT).resolve()
    if quarantine not in resolved.parents and resolved != quarantine:
        raise QuarantinePathError(
            f"{QuarantinePathError.typed_reason}: {path} resolves outside "
            f"{_QUARANTINE_ROOT}; NCU-derived count labels are quarantined."
        )


def count_metric_identifiers() -> tuple[str, ...]:
    """Every registry replay-count unit-tagged 'count' metric available on
    sm_86 — the capture target set is derived from the registry so the
    registry/probe lockstep governs what gets captured."""
    from metric_registry import load_registry

    out = []
    for e in load_registry().entries:
        if (
            e.metric_class == "replay_count"
            and e.carrier == "unit_tagged_metric"
            and e.unit == "count"
            and e.availability == "available"
            and (e.per_arch_availability or {}).get("sm_86") == "available"
        ):
            out.append(e.ncu_metric)
    return tuple(sorted(out))


def assert_targets_probe_available(metrics: tuple[str, ...]) -> None:
    """Fail closed unless every capture target is confirmed by the
    committed metric-availability probe (no head/label may target an
    unprobed metric)."""
    from probe_ncu_metric_availability import (
        assert_target_metric_available,
        load_probe,
    )

    probe = load_probe()
    for m in metrics:
        assert_target_metric_available(probe, m)


def _variant_fields(cell: dict) -> dict:
    out: dict[str, Any] = {}
    if cell.get("register_cap") is not None:
        out["register_cap"] = cell["register_cap"]
    if cell.get("variant_id") is not None:
        out["variant_id"] = cell["variant_id"]
    return out


def _binary_path_for(slug: str, expected_sha256: str | None = None) -> Path | None:
    """Resolve the on-disk binary for ``slug``, sha-first when a pin is
    supplied: build-variant cells (e.g. register-cap spill variants under
    ``<slug>/spill_variants/``) pin a sha256 that only their own variant
    binary carries, so candidate ``*_run*`` files under the slug's tree
    are matched BY PIN before falling back to the base path. A pin that
    matches nothing returns the base path (the caller types the mismatch
    against that path's sha)."""
    base: Path | None = None
    for d in _BINARY_DIRS:
        slug_dir = REPO_ROOT / d / slug
        p = slug_dir / f"{slug}_run"
        if p.is_file() and base is None:
            base = p
        if expected_sha256 and slug_dir.is_dir():
            candidates = [p] if p.is_file() else []
            candidates += sorted(slug_dir.glob("*/*_run*"))
            for cand in candidates:
                if cand.is_file() and _sha256(cand) == expected_sha256:
                    return cand
    return base


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ncu_capture_once(
    ncu: str, binary: Path, app_args: list[str], metrics: tuple[str, ...],
) -> dict[str, float]:
    """One read-only NCU metric capture -> {ncu_metric: value}. CSV to
    stdout only (no .ncu-rep)."""
    proc = subprocess.run(
        [
            ncu, "--metrics", ",".join(metrics),
            "--launch-count", "1", "--csv", str(binary), *app_args,
        ],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ncu rc={proc.returncode}: {proc.stderr[-400:]!r}")
    rows = [ln for ln in proc.stdout.splitlines() if ln.startswith('"')]
    reader = csv.DictReader(io.StringIO("\n".join(rows)))
    out: dict[str, float] = {}
    for row in reader:
        name = row.get("Metric Name")
        value = row.get("Metric Value")
        if name in metrics and value:
            out[name] = float(value.replace(",", ""))
    return out


def capture_cell_repeats(
    ncu: str,
    binary: Path,
    repeats: int,
    app_args: list[str],
    metrics: tuple[str, ...],
) -> tuple[dict[int, list[dict]], dict[str, list[float]]]:
    """Capture ``repeats`` repeats. Returns ``(by_repeat, raw_values)``:
    ``by_repeat`` is shaped exactly as the real aggregator gate consumes;
    ``raw_values`` keeps the per-metric raw repeat values for the
    integer-stability admission."""
    from metric_registry import load_registry

    registry = load_registry()
    by_repeat: dict[int, list[dict]] = {}
    raw_values: dict[str, list[float]] = {m: [] for m in metrics}
    for r in range(repeats):
        captured = _ncu_capture_once(ncu, binary, app_args, metrics)
        utm: list[dict] = []
        for ncu_metric in metrics:
            if ncu_metric not in captured:
                continue
            entry = registry.get_by_ncu_metric(ncu_metric)
            if entry is None:
                continue
            value = float(captured[ncu_metric])
            raw_values[ncu_metric].append(value)
            utm.append({
                "ncu_metric": ncu_metric,
                "unit": entry.unit,
                "value": value,
            })
        by_repeat[r] = [{"unit_tagged_metrics": utm}]
    return by_repeat, raw_values


def admit_integer_stable_counts(
    *,
    gated: list[dict],
    raw_values: dict[str, list[float]],
    metrics: tuple[str, ...],
    repeats: int,
    audit: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """The integer-stability admission (applied ON TOP of the real
    registry-rule gate). Returns ``(admitted_entries, metric_exclusions)``.

    Per metric: every repeat must be present and integral, and all repeats
    must agree exactly. A disagreeing counter is typed
    ``label_count_unstable`` with the raw repeat values recorded — never
    averaged. The admitted entry comes from the REAL gate output, whose
    mean equals the agreed integer and whose dispersion is 0.0 by
    construction.

    When ``audit`` is supplied, a per-metric stability record (raw repeat
    values, integer spread, integrality, admission outcome) is written
    into it — the committed audit artifact proving every admitted count
    label had exact integer repeat agreement.
    """
    gated_by_metric = {m["ncu_metric"]: m for m in gated}
    admitted: list[dict] = []
    excluded: list[dict] = []
    for metric in metrics:
        values = raw_values.get(metric, [])
        if audit is not None:
            audit[metric] = {
                "repeat_values": list(values),
                "n_repeats_observed": len(values),
                "all_integral": bool(values)
                and all(v == int(v) for v in values),
                "integer_spread": (
                    (max(values) - min(values)) if values else None
                ),
                "admitted": False,  # flipped below on admission
            }
        if len(values) < repeats:
            excluded.append({
                "ncu_metric": metric,
                "typed_reason": "count_capture_incomplete",
                "detail": (
                    f"{len(values)}/{repeats} repeats returned a value"
                ),
                "repeat_values": list(values),
            })
            continue
        if any(v != int(v) for v in values):
            excluded.append({
                "ncu_metric": metric,
                "typed_reason": "count_value_not_integral",
                "detail": "a count metric returned non-integral values",
                "repeat_values": list(values),
            })
            continue
        lo, hi = min(values), max(values)
        if hi != lo:
            mean = sum(values) / len(values)
            excluded.append({
                "ncu_metric": metric,
                "typed_reason": "label_count_unstable",
                "detail": (
                    "repeats disagree (integer-stability admission "
                    "requires exact agreement); never averaged"
                ),
                "repeat_values": list(values),
                "integer_spread": hi - lo,
                "relative_spread": (hi - lo) / abs(mean) if mean else None,
            })
            continue
        entry = gated_by_metric.get(metric)
        if entry is None:
            excluded.append({
                "ncu_metric": metric,
                "typed_reason": "count_capture_incomplete",
                "detail": (
                    "metric missing from the aggregator gate output "
                    "despite stable repeats"
                ),
                "repeat_values": list(values),
            })
            continue
        admitted.append(entry)
        if audit is not None:
            audit[metric]["admitted"] = True
    return admitted, excluded


def splice_count_entries(cell: dict, admitted_entries: list[dict]) -> None:
    """Additive in-place splice: existing unit_tagged_metrics entries
    (timing) are preserved; entries for the same ncu_metric are replaced
    (idempotent re-capture); the cell's keys stay sorted so untouched cells
    remain byte-identical under the store's indent=2/no-sort_keys dump."""
    fresh_ids = {e["ncu_metric"] for e in admitted_entries}
    existing = [
        e for e in (cell.get("unit_tagged_metrics") or [])
        if e["ncu_metric"] not in fresh_ids
    ]
    # Canonicalize entry key order (sorted) so a re-capture that lands
    # identical values is byte-identical on disk (idempotency proof).
    normalized = [
        {k: e[k] for k in sorted(e)} for e in admitted_entries
    ]
    cell["unit_tagged_metrics"] = sorted(
        existing + normalized, key=lambda m: m["ncu_metric"]
    )
    reordered = {k: cell[k] for k in sorted(cell)}
    cell.clear()
    cell.update(reordered)


def capture_and_land(
    *,
    label_admitted_path: Path,
    ncu: str,
    repeats: int,
    arch: str = _ARCH,
    metrics: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], list[dict], dict[str, Any]]:
    """Capture counts for each admitted ``arch`` cell, gate via the real
    aggregator machinery + the integer-stability admission, splice in
    place. Returns ``(updated_admitted_dict, exclusions, stats)``."""
    from repeatability_aggregator import (
        DEFAULT_REPEATS_REQUIRED,
        _aggregate_unit_tagged_metrics,
    )

    if not Path(ncu).is_file():
        raise FileNotFoundError(f"ncu not found at {ncu}")
    if metrics is None:
        metrics = count_metric_identifiers()
    assert_targets_probe_available(metrics)

    admitted = json.loads(label_admitted_path.read_text(encoding="utf-8"))
    exclusions: list[dict] = []
    n_landed_cells = 0
    n_landed_entries = 0
    captured_cells: list[str] = []
    audit_cells: list[dict] = []
    for cell in admitted.get("cells", []):
        if cell.get("arch") != arch:
            continue
        slug = cell["slug"]
        shape_point = cell["shape_point"]
        variant_id = cell_variant_id(cell)
        key = f"{slug}@{shape_point}@{variant_id}" if variant_id else (
            f"{slug}@{shape_point}"
        )
        vfields = _variant_fields(cell)
        binary = _binary_path_for(slug, cell.get("binary_sha256"))
        if binary is None:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "count_binary_absent",
                "detail": f"no runnable binary on disk for {slug}",
                **vfields,
            })
            continue
        disk_sha = _sha256(binary)
        if disk_sha != cell.get("binary_sha256"):
            unstable = _capture_unstable_record(slug, shape_point)
            if unstable is not None:
                # Self-contained final semantics: the cell is documented
                # capture-unstable in the committed re-admission manifest
                # (no trustworthy fresh labels exist at any available
                # binary), so consumers read the terminal typed reason +
                # evidence directly from this artifact.
                exclusions.append({
                    "slug": slug, "shape_point": shape_point,
                    "typed_reason": "count_label_capture_unstable",
                    "underlying_reason": "count_binary_sha256_mismatch",
                    "detail": (
                        f"on-disk binary sha256 {disk_sha[:12]} != admitted "
                        f"binary_sha256 "
                        f"{str(cell.get('binary_sha256'))[:12]}; the cell "
                        "is capture-unstable per the re-admission manifest "
                        "(the admitted row keeps its historical identity)"
                    ),
                    "capture_unstable_evidence": dict(
                        unstable,
                        manifest_path=DEFAULT_READMISSION_MANIFEST,
                    ),
                    **vfields,
                })
            else:
                exclusions.append({
                    "slug": slug, "shape_point": shape_point,
                    "typed_reason": "count_binary_sha256_mismatch",
                    "detail": (
                        f"on-disk binary sha256 {disk_sha[:12]} != admitted "
                        f"binary_sha256 "
                        f"{str(cell.get('binary_sha256'))[:12]}"
                    ),
                    **vfields,
                })
            continue
        try:
            app_args = _app_args_for_cell(slug=slug, shape_point=shape_point)
        except ValueError as exc:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "count_app_arg_resolution_failed",
                "detail": str(exc)[:200],
                **vfields,
            })
            continue
        try:
            by_repeat, raw_values = capture_cell_repeats(
                ncu, binary, repeats, app_args, metrics
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "count_capture_failed",
                "detail": str(exc)[:200],
                **vfields,
            })
            continue
        gated = _aggregate_unit_tagged_metrics(
            by_repeat, DEFAULT_REPEATS_REQUIRED
        )
        gated_counts = [m for m in gated if m["ncu_metric"] in metrics]
        metric_audit: dict[str, dict] = {}
        admitted_entries, metric_exclusions = admit_integer_stable_counts(
            gated=gated_counts,
            raw_values=raw_values,
            metrics=metrics,
            repeats=repeats,
            audit=metric_audit,
        )
        audit_cells.append({
            "slug": slug,
            "shape_point": shape_point,
            **vfields,
            "binary_sha256": cell.get("binary_sha256"),
            "metrics": metric_audit,
        })
        for ex in metric_exclusions:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                **ex, **vfields,
            })
        if admitted_entries:
            splice_count_entries(cell, admitted_entries)
            n_landed_cells += 1
            n_landed_entries += len(admitted_entries)
            captured_cells.append(key)
            print(
                f"  [{key}] landed {len(admitted_entries)} count metrics"
                + (
                    f"; {len(metric_exclusions)} typed"
                    if metric_exclusions else ""
                ),
                file=sys.stderr,
            )
    stats = {
        "n_landed_cells": n_landed_cells,
        "n_landed_entries": n_landed_entries,
        "captured_cells": captured_cells,
        "metrics": list(metrics),
        "audit_cells": audit_cells,
    }
    print(
        f"[capture_counts] landed {n_landed_entries} count entries on "
        f"{n_landed_cells} {arch} cells; {len(exclusions)} typed exclusions",
        file=sys.stderr,
    )
    return admitted, exclusions, stats


def build_governance_artifact(
    *,
    stats: dict[str, Any],
    repeats: int,
    arch: str = _ARCH,
) -> dict[str, Any]:
    """The sealed-holdout governance trail for the count-label capture.

    Capturing count labels on holdout cells is permitted — labels are
    ground truth, not model fit. The trail records that the admission rule
    has no tunable parameters (so no label value influenced any
    configuration choice) and that holdout SCORING touches remain governed
    by the digest-keyed touch ledger (zero at capture time)."""
    split = json.loads(
        (REPO_ROOT / _COHORT_SPLIT).read_text(encoding="utf-8")
    )
    holdout_keys = sorted(
        f"{c['slug']}@{c['shape_point']}"
        for c in split["splits"]["holdout"]["cells"]
    )
    captured = set(stats.get("captured_cells", []))
    holdout_captured = sorted(
        k for k in holdout_keys
        if any(c == k or c.startswith(k + "@") for c in captured)
    )
    return {
        "schema_version": 1,
        "artifact_kind": "count_label_capture_governance",
        "arch": arch,
        "sealed_split": {
            "path": _COHORT_SPLIT,
            "cohort_manifest_digest": split.get("cohort_manifest_digest"),
            "frozen_at_commit": split.get("frozen_at_commit"),
        },
        "holdout_capture_policy": (
            "capture-permitted: count labels are ground truth, not model "
            "fit; producer/model tuning and threshold selection use only "
            "calibration/validation cells or synthetic fixtures"
        ),
        "holdout_cells_captured": holdout_captured,
        "holdout_scoring_touches": 0,
        "holdout_touch_ledger": _TOUCH_LEDGER,
        "tuning_trail": {
            "admission_rule": "exact_integer_stability_over_n_repeats",
            "tunable_parameters": [],
            "label_values_consumed_for_tuning": False,
            "scope_statement": (
                "Admission requires bitwise-equal integral repeats; the "
                "rule has no tunable thresholds, so no calibration, "
                "validation, or holdout label value influenced any "
                "configuration choice."
            ),
        },
        "n_repeats": repeats,
        "captured_metrics": stats.get("metrics", []),
        "n_landed_cells": stats.get("n_landed_cells", 0),
        "n_landed_entries": stats.get("n_landed_entries", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-admitted", default=DEFAULT_LABEL_ADMITTED)
    parser.add_argument("--ncu", default=DEFAULT_NCU)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--arch", default=_ARCH)
    parser.add_argument("--exclusions-out", default=DEFAULT_EXCLUSIONS_OUT)
    parser.add_argument("--governance-out", default=DEFAULT_GOVERNANCE_OUT)
    parser.add_argument("--audit-out", default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    label_path = Path(args.label_admitted)
    exclusions_path = Path(args.exclusions_out)
    governance_path = Path(args.governance_out)
    audit_path = Path(args.audit_out)
    # Rule-1 quarantine: NCU-derived outputs exist only under
    # data/local_labels (fail-closed before any capture).
    for p in (label_path, exclusions_path, governance_path, audit_path):
        assert_quarantine_path(p)

    if args.dry_run:
        sys.stderr.write("[capture_counts] --dry-run: no files written\n")
        return 0

    try:
        admitted, exclusions, stats = capture_and_land(
            label_admitted_path=label_path,
            ncu=args.ncu,
            repeats=args.repeats,
            arch=args.arch,
        )
    except FileNotFoundError as exc:
        sys.stderr.write(
            f"count_capture_unavailable: {exc} (no write performed)\n"
        )
        return 3

    # indent=2 + NO sort_keys matches the committed store's formatting so
    # untouched cells stay byte-identical (frozen L2 inputs unchanged).
    label_path.write_text(
        json.dumps(admitted, indent=2) + "\n", encoding="utf-8",
    )
    excl_artifact = {
        "schema_version": 1,
        "artifact_kind": "count_label_exclusions",
        "arch": args.arch,
        "n_exclusions": len(exclusions),
        "exclusions": sorted(
            exclusions,
            key=lambda e: (
                e["slug"], e["shape_point"], e.get("variant_id") or "",
                e.get("ncu_metric") or "",
            ),
        ),
    }
    exclusions_path.write_text(
        json.dumps(excl_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    governance = build_governance_artifact(
        stats=stats, repeats=args.repeats, arch=args.arch,
    )
    governance_path.write_text(
        json.dumps(governance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_cells = stats.get("audit_cells", [])
    audit_artifact = {
        "schema_version": 1,
        "artifact_kind": "count_label_capture_audit",
        "arch": args.arch,
        "n_repeats": args.repeats,
        "n_cells": len(audit_cells),
        "note": (
            "Per-cell per-metric raw repeat values + integer-stability "
            "statistics for the replay-count capture. Every admitted "
            "count label must show all_integral=true and "
            "integer_spread=0.0 over n_repeats repeats; admitted=false "
            "rows are the typed per-metric exclusions."
        ),
        "cells": sorted(
            audit_cells,
            key=lambda c: (
                c["slug"], c["shape_point"], c.get("variant_id") or "",
            ),
        ),
    }
    audit_path.write_text(
        json.dumps(audit_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
