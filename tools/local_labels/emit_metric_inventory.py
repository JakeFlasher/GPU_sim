"""Emit the metric-ontology inventory artifact from the committed registry.

This module flattens the metric registry into one inventory row per metric,
recording the ontology coordinates a reader needs to reason about ground-truth
coverage WITHOUT re-parsing the registry's nested observation/composition
blocks:

- ``key`` / ``ncu_metric`` — the registry key and canonical Nsight Compute id.
- ``metric_class`` / ``risk_tier`` — the ontology coordinates.
- ``primitive_or_derived`` — ``derived`` for derived_ratio / derived_pct_peak,
  else ``primitive``.
- ``sm_86_availability`` / ``sm_120_availability`` — the per-arch status from
  the registry's per_arch_availability map (falling back to the legacy
  single-column availability for sm_86 when the map is absent on a pre-v2
  snapshot).
- ``ground_truth_source`` — ``ncu_counter_label`` for label_only metrics,
  ``static_feature`` for feature_eligible (static, launch-config-derivable)
  metrics.
- ``typed_unavailable_reason`` — a short reason string when the metric is not
  reachable on sm_86 (probe_pending / excluded / typed_unavailable), else
  ``None``.

``build_inventory(registry)`` returns the list of rows; the ``__main__`` entry
point loads the committed registry and writes the artifact to
``data/local_labels/metric_ontology_inventory.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metric_registry import MetricEntry, MetricRegistry, load_registry

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_PATH = (
    _REPO_ROOT / "data" / "local_labels" / "metric_ontology_inventory.json"
)


def _ground_truth_source(deployability: str) -> str:
    if deployability == "feature_eligible":
        return "static_feature"
    return "ncu_counter_label"


def _typed_unavailable_reason(sm_86_availability: str) -> str | None:
    """A short reason string when the metric is not host-available on sm_86."""
    reasons = {
        "available": None,
        "probe_pending": (
            "sm_86_probe_did_not_confirm_canonical_identifier_on_this_card"
        ),
        "excluded": "sm_86_probe_found_metric_unavailable_or_renamed",
        "unresolved": "no_settled_canonical_identifier_yet",
        "typed_unavailable": "no_ground_truth_reachable_on_this_arch",
        "sm_120_deferred_no_live_evidence": "no_live_evidence_for_this_arch",
        "requires_runtime_carveout": "requires_runtime_shared_mem_carveout",
    }
    return reasons.get(sm_86_availability, f"unmapped_status:{sm_86_availability}")


def _predictor_composition(e: MetricEntry) -> tuple[str, str | None]:
    """Whether a derived metric carries a real predictor-side composition from
    registered primitives. A derived row whose composition names no
    dependencies and no device constants is a label-only/sentinel row (e.g.
    ``label_only_no_predictor_composition`` /
    ``speed_of_light_requires_downstream_calibration``) with NO predictor
    composition — surfaced explicitly so a reader does not mistake the
    ground-truth label for a composed prediction. Returns
    ``(status, formula_or_none)``."""
    if not e.is_derived:
        return "not_applicable", None
    comp = e.composition or {}
    formula = comp.get("formula")
    deps = comp.get("dependencies") or []
    consts = comp.get("device_constants") or []
    if deps or consts:
        return "composed", formula
    return "no_predictor_composition", formula


def build_inventory(registry: MetricRegistry) -> list[dict[str, Any]]:
    """One inventory row per registry metric."""
    rows: list[dict[str, Any]] = []
    for e in registry.entries:
        per_arch = e.per_arch_availability or {}
        sm_86_availability = per_arch.get("sm_86", e.availability)
        sm_120_availability = per_arch.get(
            "sm_120", "sm_120_deferred_no_live_evidence"
        )
        composition_status, composition_formula = _predictor_composition(e)
        rows.append(
            {
                "key": e.key,
                "ncu_metric": e.ncu_metric,
                "metric_class": e.metric_class,
                "risk_tier": e.risk_tier,
                "primitive_or_derived": "derived" if e.is_derived else "primitive",
                "sm_86_availability": sm_86_availability,
                "sm_120_availability": sm_120_availability,
                "ground_truth_source": _ground_truth_source(e.deployability),
                "typed_unavailable_reason": _typed_unavailable_reason(
                    sm_86_availability
                ),
                "predictor_composition": composition_status,
                "composition_formula": composition_formula,
            }
        )
    return rows


def emit(path: Path | None = None) -> Path:
    registry = load_registry()
    rows = build_inventory(registry)
    out_path = path if path is not None else _ARTIFACT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "registry_version": registry.registry_version,
        "label_schema_version": registry.label_schema_version,
        "description": (
            "Flat metric-ontology inventory: one row per registry metric with "
            "its metric_class, risk_tier, primitive/derived kind, per-arch "
            "availability, ground-truth source, and (when not host-available on "
            "sm_86) a typed-unavailable reason. Generated from "
            "tools/local_labels/metric_registry.json by "
            "tools/local_labels/emit_metric_inventory.py."
        ),
        "metric_count": len(rows),
        "metrics": rows,
    }
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return out_path


if __name__ == "__main__":
    written = emit()
    print(f"wrote {written} ({len(build_inventory(load_registry()))} rows)")
