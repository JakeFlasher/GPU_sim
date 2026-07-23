"""No-deployability-leakage guard.

Theoretical occupancy / static launch-limit metrics are static-derivable and so
MAY appear as predictor input features. Achieved (counter-derived) occupancy
(`sm__warps_active.avg.pct_of_peak_sustained_active`) and IPC
(`sm__inst_executed.avg.per_cycle_elapsed`) are LABEL-ONLY: they may be captured
as labels/diagnostics but must NEVER enter a deployable inference feature
surface, because the deployed predictor runs in environments without those
admin-required counters. This guard enforces that contract from the committed
metric registry's `deployability` field, so the policy has a single source of
truth.

A "deployable inference feature surface" is any collection of feature field
names fed to a head at inference time (e.g. the canonical feature vector). The
guard rejects any feature name that resolves — by label-style field name OR by
canonical NCU metric id — to a `label_only` registry metric. `feature_eligible`
metrics (theoretical occupancy) are explicitly allowed.
"""

from __future__ import annotations

from typing import Iterable

from metric_registry import MetricRegistry, load_registry


class DeployabilityLeakageError(ValueError):
    """Raised when a deployable inference feature surface contains a label-only
    (achieved-occupancy / IPC, or any other label_only) metric."""


def label_only_feature_aliases(registry: MetricRegistry | None = None) -> dict[str, str]:
    """Map every name that would identify a ``label_only`` metric (its registry
    key, sidecar field, admitted field, and canonical NCU id) to that metric's
    key. These names are FORBIDDEN as deployable inference features."""
    reg = registry or load_registry()
    aliases: dict[str, str] = {}
    for e in reg.entries:
        if e.deployability != "label_only":
            continue
        for name in (e.key, e.sidecar_field, e.admitted_field, e.ncu_metric):
            aliases[name] = e.key
    return aliases


def feature_eligible_aliases(registry: MetricRegistry | None = None) -> dict[str, str]:
    """Map every name identifying a ``feature_eligible`` metric (theoretical
    occupancy / static launch-limit) to its key. These ARE allowed as features."""
    reg = registry or load_registry()
    aliases: dict[str, str] = {}
    for e in reg.entries:
        if e.deployability != "feature_eligible":
            continue
        for name in (e.key, e.sidecar_field, e.admitted_field, e.ncu_metric):
            aliases[name] = e.key
    return aliases


def scan_features_for_leakage(
    feature_fields: Iterable[str], registry: MetricRegistry | None = None
) -> list[dict]:
    """Return one record per leaked feature: a feature name that resolves to a
    ``label_only`` metric. Empty list == no leakage."""
    reg = registry or load_registry()
    forbidden = label_only_feature_aliases(reg)
    leaks: list[dict] = []
    for field in feature_fields:
        if field in forbidden:
            leaks.append({"feature_field": field, "label_only_metric_key": forbidden[field]})
    return leaks


def assert_no_deployability_leakage(
    feature_fields: Iterable[str], registry: MetricRegistry | None = None
) -> None:
    """Fail closed if any feature name resolves to a label_only metric."""
    leaks = scan_features_for_leakage(feature_fields, registry)
    if leaks:
        detail = ", ".join(
            f"{lk['feature_field']!r} -> label_only metric {lk['label_only_metric_key']!r}"
            for lk in leaks
        )
        raise DeployabilityLeakageError(
            f"deployable inference feature surface contains label-only "
            f"(achieved-occupancy / IPC) metrics: {detail}. Achieved occupancy "
            f"and IPC are counter-derived and may be captured only as labels, "
            f"never as deployable inference inputs."
        )
