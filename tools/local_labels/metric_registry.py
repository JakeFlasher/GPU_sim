"""Loader and resolver for the committed local-label metric registry.

The registry (``metric_registry.json``, alongside this module) is the single
source of truth that binds every ground-truth target metric to:

- its canonical Nsight Compute metric identifier (``ncu_metric``);
- the label-style field names it is carried under in the per-launch sidecar and
  the per-cell admitted record (``sidecar_field`` / ``admitted_field``);
- its ``unit`` (one of ``pct``, ``count``, ``ns``, ``cycles``);
- its valid range, parser transform, aggregation rule, per-unit repeatability
  rule, nullable policy, and deployability (``label_only`` vs
  ``feature_eligible``);
- its carrier (a historical ``flat_field`` or the schema-v5
  ``unit_tagged_metric`` block).

The percentage family (hit rates) keeps the historical ``flat_field`` carrier
and the ``percent_to_fraction`` transform; every non-percentage family
(``count`` / ``ns`` / ``cycles``) is carried under the ``unit_tagged_metrics``
block introduced at label schema version 5. Consumers (the parser, the
aggregator, the importer) resolve a metric's unit and transform here rather
than hard-coding a percentage assumption, so a count/ns/cycles value can never
silently travel the percentage ``/100`` path.

Public surface:

- ``parse_registry(raw)`` -> parsed registry from caller-supplied data without
  consulting the on-disk active registry.
- ``load_registry()`` -> parsed registry dict (cached per path).
- ``MetricRegistry`` dataclass wrapping the parsed registry with resolver
  helpers (by key, by ncu_metric, by sidecar/admitted field).
- ``MetricEntry`` dataclass for one metric row.
- ``UNITS`` / ``PARSER_TRANSFORMS`` / ``REPEATABILITY_RULES`` constant tuples.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).resolve().parent / "metric_registry.json"

# Mirror of the registry's closed vocabularies. These are asserted equal to the
# on-disk registry by test_metric_registry.py so a drift fails fast.
UNITS: tuple[str, ...] = ("pct", "count", "ns", "cycles")
PARSER_TRANSFORMS: tuple[str, ...] = ("percent_to_fraction", "identity_nonneg")
REPEATABILITY_RULES: tuple[str, ...] = (
    "stddev_abs",
    "coefficient_of_variation",
    "relative_tolerance",
)
DEPLOYABILITY_KINDS: tuple[str, ...] = ("label_only", "feature_eligible")
CARRIER_KINDS: tuple[str, ...] = ("flat_field", "unit_tagged_metric")
# Availability tokens an entry may declare. ``available`` = confirmed on this
# card; ``probe_pending`` = awaiting the metric-availability probe;
# ``unresolved`` = no settled canonical identifier yet (no head may target it);
# ``excluded`` = the probe found it unavailable/renamed on this card.
AVAILABILITY_TOKENS: tuple[str, ...] = (
    "available",
    "probe_pending",
    "unresolved",
    "excluded",
)

# Metric-ontology vocabulary (registry_version 2+). Each target metric declares
# a metric_class describing HOW it is produced and a risk_tier describing how
# hard/uncertain it is to reproduce without privileged counters:
#   - static          (T0): launch/resource arithmetic, no GPU run.
#   - replay_count    (T1): instruction/request/sector counts from trace replay.
#   - cache_stream    (T2): cross-level miss-stream / reuse-distance modeling.
#   - time_clock      (T3): DVFS/scheduler-bound timing & occupancy.
#   - derived_ratio        : composed ratio of primitives (e.g. hit_rate, IPC).
#   - derived_pct_peak     : composed percentage-of-peak of primitives.
#   - typed_unavailable    : no ground truth / not reachable on this arch.
# These fields are OPTIONAL at the parser layer so older registry snapshots and
# additive future entries keep validating; when present they are vocabulary-
# checked. The composition DAG / resolver consumes ncu_observation_contract and
# composition; this loader only stores+vocab-validates them.
METRIC_CLASSES: tuple[str, ...] = (
    "static",
    "replay_count",
    "cache_stream",
    "time_clock",
    "derived_ratio",
    "derived_pct_peak",
    "typed_unavailable",
)
RISK_TIERS: tuple[str, ...] = ("T0", "T1", "T2", "T3")

# Registry versions >= this value require the metric-ontology fields
# (metric_class, risk_tier, ncu_observation_contract, per_arch_availability)
# on every entry, and a composition block on every derived entry. Older
# snapshots (registry_version < this) keep those fields OPTIONAL so they
# continue to validate (backward compatibility).
STRICT_ONTOLOGY_REGISTRY_VERSION: int = 2

# Required sub-keys of an entry's ncu_observation_contract (registry_version>=2).
NCU_OBSERVATION_CONTRACT_KEYS: tuple[str, ...] = (
    "counting_unit",
    "rollup",
    "numerator",
    "denominator",
    "op_classes",
    "unsupported_op_behavior",
)
# Required sub-keys of a derived entry's composition block.
COMPOSITION_KEYS: tuple[str, ...] = (
    "formula",
    "dependencies",
    "device_constants",
)
# Per-arch availability map must declare a status for both local devices.
PER_ARCH_AVAILABILITY_KEYS: tuple[str, ...] = ("sm_86", "sm_120")

# The label schema version this registry is aligned with. Bumped in lockstep
# with the sidecar/admitted schema_version.
LABEL_SCHEMA_VERSION: int = 5


class MetricRegistryError(ValueError):
    """Raised when the registry is malformed or a lookup target is unknown."""


@dataclass(frozen=True)
class MetricEntry:
    """One resolved metric row from the registry."""

    key: str
    ncu_metric: str
    unit: str
    family: str
    carrier: str
    sidecar_field: str
    admitted_field: str
    valid_min: float
    valid_max: float | None
    parser_transform: str
    aggregation: str
    repeatability_rule: str
    nullable: bool
    deployability: str
    availability: str
    cache_layer: str | None = None
    traffic_subset: str | None = None
    notes: str = ""
    metric_class: str | None = None
    risk_tier: str | None = None
    ncu_observation_contract: dict[str, Any] | None = None
    composition: dict[str, Any] | None = None
    per_arch_availability: dict[str, str] | None = None

    @property
    def is_derived(self) -> bool:
        return self.metric_class in ("derived_ratio", "derived_pct_peak")

    @property
    def is_percentage(self) -> bool:
        return self.parser_transform == "percent_to_fraction"

    @property
    def is_flat_field(self) -> bool:
        return self.carrier == "flat_field"

    @property
    def is_unit_tagged(self) -> bool:
        return self.carrier == "unit_tagged_metric"

    @property
    def is_feature_eligible(self) -> bool:
        return self.deployability == "feature_eligible"

    @property
    def is_static(self) -> bool:
        return self.metric_class == "static"

    @property
    def is_label_only(self) -> bool:
        return self.deployability == "label_only"


@dataclass(frozen=True)
class MetricRegistry:
    """In-memory wrapper over the parsed registry with resolver helpers."""

    registry_version: int
    label_schema_version: int
    repeatability_thresholds: dict[str, float]
    entries: tuple[MetricEntry, ...]

    def _index(self, attr: str) -> dict[str, MetricEntry]:
        out: dict[str, MetricEntry] = {}
        for e in self.entries:
            out[getattr(e, attr)] = e
        return out

    def by_key(self, key: str) -> MetricEntry:
        try:
            return self._index("key")[key]
        except KeyError as exc:
            raise MetricRegistryError(
                f"no registry metric with key {key!r}"
            ) from exc

    def by_ncu_metric(self, ncu_metric: str) -> MetricEntry:
        try:
            return self._index("ncu_metric")[ncu_metric]
        except KeyError as exc:
            raise MetricRegistryError(
                f"no registry metric with ncu_metric {ncu_metric!r}"
            ) from exc

    def get_by_ncu_metric(self, ncu_metric: str) -> MetricEntry | None:
        return self._index("ncu_metric").get(ncu_metric)

    def by_sidecar_field(self, field: str) -> MetricEntry | None:
        return self._index("sidecar_field").get(field)

    def by_admitted_field(self, field: str) -> MetricEntry | None:
        return self._index("admitted_field").get(field)

    def unit_tagged_entries(self) -> tuple[MetricEntry, ...]:
        return tuple(e for e in self.entries if e.is_unit_tagged)

    def flat_field_entries(self) -> tuple[MetricEntry, ...]:
        return tuple(e for e in self.entries if e.is_flat_field)

    def feature_eligible_ncu_metrics(self) -> frozenset[str]:
        return frozenset(e.ncu_metric for e in self.entries if e.is_feature_eligible)

    def label_only_ncu_metrics(self) -> frozenset[str]:
        return frozenset(
            e.ncu_metric for e in self.entries if e.deployability == "label_only"
        )

    def threshold_for(self, entry: MetricEntry) -> float:
        return self.repeatability_thresholds[entry.repeatability_rule]

    def entries_by_risk_tier(self, tier: str) -> tuple[MetricEntry, ...]:
        return tuple(e for e in self.entries if e.risk_tier == tier)

    def entries_by_metric_class(self, mclass: str) -> tuple[MetricEntry, ...]:
        return tuple(e for e in self.entries if e.metric_class == mclass)

    def label_only_aliases(self) -> dict[str, str]:
        """Every name that identifies a ``label_only`` metric (its registry key,
        sidecar field, admitted field, and canonical NCU id) mapped to that
        metric's key. These names are FORBIDDEN as deployable inference
        features."""
        aliases: dict[str, str] = {}
        for e in self.entries:
            if e.deployability != "label_only":
                continue
            for name in (e.key, e.sidecar_field, e.admitted_field, e.ncu_metric):
                aliases[name] = e.key
        return aliases

    def assert_no_label_leakage(self, feature_keys: set[str]) -> None:
        """No-NCU-label-leakage guard. Fail closed if any feature key resolves
        (by registry key, sidecar field, admitted field, or canonical NCU id)
        to a ``label_only`` registry entry. ``feature_eligible`` metrics (the
        static theoretical-occupancy limits) are explicitly allowed, so a
        deployable feature surface may name them without tripping this guard."""
        forbidden = self.label_only_aliases()
        leaks = sorted(
            (str(f), forbidden[f]) for f in feature_keys if f in forbidden
        )
        if leaks:
            detail = ", ".join(
                f"{field!r} -> label_only metric {key!r}" for field, key in leaks
            )
            raise MetricRegistryError(
                f"deployable feature surface leaks label-only metrics: {detail}. "
                f"label_only entries are NCU-counter ground-truth labels and may "
                f"never be admitted as deployable inference features."
            )


def _parse(raw: dict[str, Any]) -> MetricRegistry:
    if not isinstance(raw, dict):
        raise MetricRegistryError("registry root must be a JSON object")
    try:
        registry_version = int(raw["registry_version"])
        label_schema_version = int(raw["label_schema_version"])
        thresholds = dict(raw["repeatability_thresholds"])
        metrics = raw["metrics"]
    except (KeyError, TypeError, ValueError) as exc:
        raise MetricRegistryError(f"registry missing required field: {exc}") from exc

    if label_schema_version != LABEL_SCHEMA_VERSION:
        raise MetricRegistryError(
            f"registry label_schema_version {label_schema_version} != expected "
            f"{LABEL_SCHEMA_VERSION}; the registry and the JSON schemas must move "
            f"in lockstep"
        )
    if not isinstance(metrics, list) or not metrics:
        raise MetricRegistryError("registry 'metrics' must be a non-empty array")

    # registry_version >= 2 promotes the metric-ontology fields from optional to
    # required; older snapshots keep them optional (backward compatibility).
    strict_ontology = registry_version >= STRICT_ONTOLOGY_REGISTRY_VERSION

    entries: list[MetricEntry] = []
    seen_keys: set[str] = set()
    seen_ncu: set[str] = set()
    for row in metrics:
        if not isinstance(row, dict):
            raise MetricRegistryError("each registry metric must be an object")
        key = row.get("key")
        ncu_metric = row.get("ncu_metric")
        unit = row.get("unit")
        transform = row.get("parser_transform")
        rule = row.get("repeatability_rule")
        deploy = row.get("deployability")
        carrier = row.get("carrier")
        avail = row.get("availability")
        if unit not in UNITS:
            raise MetricRegistryError(f"metric {key!r}: bad unit {unit!r}")
        if transform not in PARSER_TRANSFORMS:
            raise MetricRegistryError(
                f"metric {key!r}: bad parser_transform {transform!r}"
            )
        if rule not in REPEATABILITY_RULES:
            raise MetricRegistryError(
                f"metric {key!r}: bad repeatability_rule {rule!r}"
            )
        if deploy not in DEPLOYABILITY_KINDS:
            raise MetricRegistryError(
                f"metric {key!r}: bad deployability {deploy!r}"
            )
        if carrier not in CARRIER_KINDS:
            raise MetricRegistryError(f"metric {key!r}: bad carrier {carrier!r}")
        if avail not in AVAILABILITY_TOKENS:
            raise MetricRegistryError(f"metric {key!r}: bad availability {avail!r}")
        # Unit/transform coherence: only the percentage transform is allowed for
        # the pct unit, and the percentage transform is allowed only for pct.
        if (unit == "pct") != (transform == "percent_to_fraction"):
            raise MetricRegistryError(
                f"metric {key!r}: unit {unit!r} and transform {transform!r} are "
                f"incoherent (pct<->percent_to_fraction must agree)"
            )
        if key in seen_keys:
            raise MetricRegistryError(f"duplicate registry key {key!r}")
        if ncu_metric in seen_ncu:
            raise MetricRegistryError(f"duplicate registry ncu_metric {ncu_metric!r}")
        seen_keys.add(key)
        seen_ncu.add(ncu_metric)
        if rule not in thresholds:
            raise MetricRegistryError(
                f"metric {key!r}: repeatability_rule {rule!r} has no threshold"
            )
        metric_class = row.get("metric_class")
        if strict_ontology and metric_class is None:
            raise MetricRegistryError(
                f"metric {key!r}: metric_class is required at registry_version "
                f"{registry_version}"
            )
        if metric_class is not None and metric_class not in METRIC_CLASSES:
            raise MetricRegistryError(
                f"metric {key!r}: bad metric_class {metric_class!r}"
            )
        risk_tier = row.get("risk_tier")
        if strict_ontology and risk_tier is None:
            raise MetricRegistryError(
                f"metric {key!r}: risk_tier is required at registry_version "
                f"{registry_version}"
            )
        if risk_tier is not None and risk_tier not in RISK_TIERS:
            raise MetricRegistryError(f"metric {key!r}: bad risk_tier {risk_tier!r}")
        for _dict_field in (
            "ncu_observation_contract",
            "composition",
            "per_arch_availability",
        ):
            _v = row.get(_dict_field)
            if _v is not None and not isinstance(_v, dict):
                raise MetricRegistryError(
                    f"metric {key!r}: {_dict_field} must be an object when present"
                )
        obs_contract = row.get("ncu_observation_contract")
        per_arch = row.get("per_arch_availability")
        composition = row.get("composition")
        is_derived = metric_class in ("derived_ratio", "derived_pct_peak")
        if strict_ontology:
            if obs_contract is None:
                raise MetricRegistryError(
                    f"metric {key!r}: ncu_observation_contract is required at "
                    f"registry_version {registry_version}"
                )
            missing_obs = [
                k for k in NCU_OBSERVATION_CONTRACT_KEYS if k not in obs_contract
            ]
            if missing_obs:
                raise MetricRegistryError(
                    f"metric {key!r}: ncu_observation_contract missing required "
                    f"sub-keys {missing_obs}"
                )
            if per_arch is None:
                raise MetricRegistryError(
                    f"metric {key!r}: per_arch_availability is required at "
                    f"registry_version {registry_version}"
                )
            missing_arch = [
                k for k in PER_ARCH_AVAILABILITY_KEYS if k not in per_arch
            ]
            if missing_arch:
                raise MetricRegistryError(
                    f"metric {key!r}: per_arch_availability missing required "
                    f"keys {missing_arch}"
                )
            if is_derived:
                if composition is None:
                    raise MetricRegistryError(
                        f"metric {key!r}: composition is required for derived "
                        f"metric_class {metric_class!r} at registry_version "
                        f"{registry_version}"
                    )
                missing_comp = [
                    k for k in COMPOSITION_KEYS if k not in composition
                ]
                if missing_comp:
                    raise MetricRegistryError(
                        f"metric {key!r}: composition missing required sub-keys "
                        f"{missing_comp}"
                    )
        vmax = row.get("valid_max")
        entries.append(
            MetricEntry(
                key=key,
                ncu_metric=ncu_metric,
                unit=unit,
                family=row.get("family", ""),
                carrier=carrier,
                sidecar_field=row["sidecar_field"],
                admitted_field=row["admitted_field"],
                valid_min=float(row.get("valid_min", 0.0)),
                valid_max=None if vmax is None else float(vmax),
                parser_transform=transform,
                aggregation=row.get("aggregation", ""),
                repeatability_rule=rule,
                nullable=bool(row.get("nullable", True)),
                deployability=deploy,
                availability=avail,
                cache_layer=row.get("cache_layer"),
                traffic_subset=row.get("traffic_subset"),
                notes=row.get("notes", ""),
                metric_class=metric_class,
                risk_tier=risk_tier,
                ncu_observation_contract=obs_contract,
                composition=composition,
                per_arch_availability=per_arch,
            )
        )
    return MetricRegistry(
        registry_version=registry_version,
        label_schema_version=label_schema_version,
        repeatability_thresholds={str(k): float(v) for k, v in thresholds.items()},
        entries=tuple(entries),
    )


def validate_unit_tag(
    registry: MetricRegistry, ncu_metric: str, claimed_unit: str
) -> tuple[bool, str, str]:
    """Cross-check a unit-tagged-metric record against the registry.

    Returns ``(ok, typed_reason, detail)``. Rejects an unregistered metric
    (``unit_tagged_metric_unregistered``) and a claimed unit that disagrees
    with the registry's unit for that metric
    (``unit_tagged_metric_unit_mismatch``) — the latter is exactly the
    "a count/ns/cycles value pushed through the percentage path" guard at the
    record layer.
    """
    entry = registry.get_by_ncu_metric(ncu_metric)
    if entry is None:
        return (
            False,
            "unit_tagged_metric_unregistered",
            f"unit-tagged metric {ncu_metric!r} is not in the metric registry",
        )
    if claimed_unit != entry.unit:
        return (
            False,
            "unit_tagged_metric_unit_mismatch",
            (
                f"unit-tagged metric {ncu_metric!r} carries unit "
                f"{claimed_unit!r} but the registry pins unit {entry.unit!r}"
            ),
        )
    return True, "", ""


def parse_registry(raw: Any) -> MetricRegistry:
    """Validate caller-supplied registry data without reading ambient files.

    The input is deep-copied before parsing so the returned registry does not
    retain aliases to mutable nested objects owned by the caller. This is the
    authority-bound counterpart to :func:`load_registry` for consumers that
    have already authenticated exact registry bytes.
    """
    try:
        return _parse(copy.deepcopy(raw))
    except MetricRegistryError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MetricRegistryError(f"registry is malformed: {exc}") from exc


@lru_cache(maxsize=4)
def load_registry(path: str | None = None) -> MetricRegistry:
    """Load and validate the metric registry. Cached per resolved path."""
    p = Path(path) if path is not None else _REGISTRY_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    return parse_registry(raw)


def load_registry_raw(path: str | None = None) -> dict[str, Any]:
    """Return the raw parsed registry JSON (uncached convenience)."""
    p = Path(path) if path is not None else _REGISTRY_PATH
    return json.loads(p.read_text(encoding="utf-8"))
