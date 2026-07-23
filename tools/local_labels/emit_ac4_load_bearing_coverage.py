"""Emit the load-bearing-metric slug-disjoint coverage artifact (task6).

For every LOAD-BEARING metric (risk_tier T0/T1 with metric_class in
{static, replay_count} in the committed registry), this report records how many
slug-disjoint admitted+evaluable sm_86 microbench families exercise it, and
classifies whether the metric clears the >= 3-slug-disjoint bar that guards a
load-bearing claim (the structural fix for leave-one-slug-out column collapse).
A metric whose only evaluatable cells come from a single slug is barred from a
load-bearing claim; a metric with no admitted coverage at all is typed
unavailable.

Two coverage bases, both committed and machine-readable:

  * OP-CLASS-GATED count metrics (global / local / texture / surface
    ld/st/atom/red requests and sectors): scored against the CONTRACT-LOCKED
    factorial-microbench SASS signatures in
    ``workloads/simbench_bw/verify_kernel_contracts.py::KERNEL_CONTRACT_SPEC`` --
    the authoritative, regression-locked declaration of which memory op-class
    each designed factorial microbench exercises. A slug exercises the metric
    iff its decoded contract op-class set intersects the metric's registry
    ``op_classes``. The imported real-workload cohort cells carry no
    contract-locked op-class signature; they are NOT counted toward DESIGNED
    factorial op-class coverage (they contribute to the cohort's
    leave-one-slug-out column variation but are not a contract-locked factorial
    family pinned to a specific op-class). Op-classes with zero contract-locked
    admitted coverage fail closed to typed-unavailable (never silently assumed).
    The load-only cache-op-classifier manifests are deliberately NOT used as the
    op-class basis: they witness only LOAD cache-ops, so they cannot witness
    store / atomic / reduction ops and would bias the per-op coverage.

  * LAUNCH / INSTRUCTION / AGGREGATE metrics (T0 ``occupancy_limit_*``
    [metric_class static], ``smsp__inst_executed`` [empty ``op_classes``], and
    the all-class L1 total-requests metric whose ``op_classes`` is the mixed
    aggregate vocabulary): launch-geometry / any-instruction / aggregate-L1
    properties of EVERY admitted cell -- exercised by every admitted+evaluable
    slug (no op-class gate).

The admitted+evaluable slug set is ``label_admitted.json`` (arch sm_86)
INTERSECTED with the regen ledger's ``status == "admitted"`` (bundle-present)
entries; register-cap build variants collapse to their base slug; label-only /
bundle_failed / unadmitted cells are excluded.

``build_coverage(...)`` returns the artifact; the ``__main__`` entry point writes
it to ``data/local_labels/ac4_load_bearing_metric_coverage_sm_86.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SIMBENCH = _REPO_ROOT / "workloads" / "simbench_bw"
if str(_SIMBENCH) not in sys.path:
    sys.path.insert(0, str(_SIMBENCH))

from metric_registry import MetricRegistry, load_registry  # noqa: E402

_ARCH = "sm_86"
_MIN_SLUG_DISJOINT = 3  # the >= 3 slug-disjoint families bar for task6.

_LABEL_ADMITTED = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_LEDGER = _REPO_ROOT / "data" / "local_labels" / "sm86_cohort_regen_ledger.json"
_ROSTER_PATHS = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "roster.toml",
    _REPO_ROOT / "workloads" / "portable_lcd" / "roster.toml",
)
_ARTIFACT_PATH = (
    _REPO_ROOT / "data" / "local_labels"
    / "ac4_load_bearing_metric_coverage_sm_86.json"
)

# The specific (non-aggregate) op-class tokens a count metric can be gated on.
# A metric whose op_classes are all in this set is op-class-gated; a metric with
# empty op_classes, a static class, or any generic aggregate token (e.g. the
# mixed 'local'/'atomic'/'reduction'/'surface' total-requests vocabulary) is a
# launch/instruction/aggregate metric exercised by every admitted cell.
_GATEABLE_OP_CLASSES: frozenset[str] = frozenset({
    "global_load",
    "global_store",
    "global_atomic",
    "global_reduction",
    "local_load",
    "local_store",
    "texture",
    "surface_load",
    "surface_store",
    "surface_reduction",
    "surface_atomic",
})


def _decode_opcode_to_op_class(opcode: str) -> str | None:
    """Decode a dotted SASS opcode head to a gateable memory op-class, or
    ``None`` for compute / shared / control opcodes (HMMA/IMMA/QMMA, barriers,
    etc.) that are not one of the tracked memory op-classes."""
    head = str(opcode).split(".", 1)[0].upper()
    if head == "LDGSTS":  # cp.async global->shared load reads from global memory
        return "global_load"
    if head.startswith("LDG"):
        return "global_load"
    if head.startswith("STG"):
        return "global_store"
    if head.startswith("ATOMG") or head.startswith("ATOM"):
        return "global_atomic"
    if head.startswith("REDG") or head.startswith("RED"):
        return "global_reduction"
    if head == "LDL":
        return "local_load"
    if head == "STL":
        return "local_store"
    if head.startswith("TEX") or head.startswith("TLD"):
        return "texture"
    if head.startswith("SULD"):
        return "surface_load"
    if head.startswith("SUST"):
        return "surface_store"
    if head.startswith("SURED"):
        return "surface_reduction"
    if head.startswith("SUATOM"):
        return "surface_atomic"
    return None


def _contract_op_classes_by_slug() -> dict[str, frozenset[str]]:
    """Decoded memory op-class set per slug, from the committed contract-locked
    SASS signatures. A slug present with only compute opcodes (e.g. an MMA
    dependency chain) maps to the empty set (no memory op-class)."""
    from verify_kernel_contracts import KERNEL_CONTRACT_SPEC  # noqa: E402

    out: dict[str, frozenset[str]] = {}
    for slug, opcodes in KERNEL_CONTRACT_SPEC.items():
        decoded = {
            oc for opcode in opcodes
            if (oc := _decode_opcode_to_op_class(opcode)) is not None
        }
        out[slug] = frozenset(decoded)
    return out


def _roster_family_by_slug() -> dict[str, str]:
    import tomllib

    out: dict[str, str] = {}
    for path in _ROSTER_PATHS:
        if not path.is_file():
            continue
        for row in tomllib.loads(path.read_text(encoding="utf-8")).get("rows", []):
            slug = row.get("slug")
            if slug and slug not in out:
                out[slug] = str(row.get("family", "unknown"))
    return out


def _admitted_evaluable_slugs() -> list[str]:
    """Distinct base slugs that are BOTH a live admitted sm_86 label AND an
    ``admitted``-status (bundle-present) regen-ledger entry. Register-cap build
    variants collapse to their base slug; label-only / bundle_failed /
    unadmitted cells are excluded."""
    admitted_labels = json.loads(_LABEL_ADMITTED.read_text(encoding="utf-8"))
    label_slugs = {
        c["slug"] for c in admitted_labels.get("cells", [])
        if c.get("arch") == _ARCH
    }
    ledger = json.loads(_LEDGER.read_text(encoding="utf-8"))
    evaluable: set[str] = set()
    for c in ledger.get("cells", []):
        if c.get("status") != "admitted":
            continue
        slug = c.get("slug")
        if slug in label_slugs:
            evaluable.add(slug)
    return sorted(evaluable)


def _metric_is_op_class_gated(metric_class: str, op_classes: list[str]) -> bool:
    """True iff the metric is a replay-count metric gated on specific memory
    op-classes (every op_class token is a specific gateable token). Static,
    empty-op_class, and mixed-aggregate metrics are launch/instruction/aggregate
    metrics (not op-class-gated)."""
    if metric_class != "replay_count":
        return False
    if not op_classes:
        return False
    return all(oc in _GATEABLE_OP_CLASSES for oc in op_classes)


def _status_for_count(count: int) -> str:
    if count >= _MIN_SLUG_DISJOINT:
        return "load_bearing_eligible"
    if count == 0:
        return "typed_unavailable_no_coverage"
    return "below_slug_disjoint_threshold"


def build_coverage(registry: MetricRegistry) -> dict[str, Any]:
    eval_slugs = _admitted_evaluable_slugs()
    family_by_slug = _roster_family_by_slug()
    contract_oc = _contract_op_classes_by_slug()

    contract_locked = sorted(s for s in eval_slugs if s in contract_oc)
    imported_no_contract = sorted(s for s in eval_slugs if s not in contract_oc)

    def families_of(slugs: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in slugs:
            out.setdefault(family_by_slug.get(s, "unknown"), []).append(s)
        return {k: out[k] for k in sorted(out)}

    metrics_rows: list[dict[str, Any]] = []
    for e in registry.entries:
        if e.risk_tier not in ("T0", "T1"):
            continue
        if e.metric_class not in ("static", "replay_count"):
            continue
        op_classes = list((e.ncu_observation_contract or {}).get("op_classes", []))
        gated = _metric_is_op_class_gated(e.metric_class, op_classes)
        if not gated:
            exercising = sorted(eval_slugs)
            basis = "all_admitted_cohort"
        else:
            wanted = set(op_classes)
            exercising = sorted(
                s for s in contract_locked
                if contract_oc.get(s, frozenset()) & wanted
            )
            basis = "contract_locked_factorial_signature"
        count = len(exercising)
        metrics_rows.append({
            "key": e.key,
            "ncu_metric": e.ncu_metric,
            "metric_class": e.metric_class,
            "risk_tier": e.risk_tier,
            "registry_op_classes": op_classes,
            "op_class_gated": gated,
            "coverage_basis": basis,
            "exercising_slugs": exercising,
            "exercising_families": families_of(exercising),
            "slug_disjoint_count": count,
            "min_slug_disjoint_families": _MIN_SLUG_DISJOINT,
            "status": _status_for_count(count),
        })
    metrics_rows.sort(key=lambda r: r["key"])
    return {
        "schema_version": 1,
        "artifact_kind": "load_bearing_metric_slug_disjoint_coverage",
        "arch": _ARCH,
        "registry_version": registry.registry_version,
        "description": (
            "Per load-bearing metric (risk_tier T0/T1, metric_class "
            "static/replay_count): the slug-disjoint admitted+evaluable sm_86 "
            "coverage and whether it clears the >= 3-slug-disjoint bar that "
            "guards a load-bearing claim. Op-class-gated count metrics are "
            "scored against the contract-locked factorial-microbench SASS "
            "signatures (KERNEL_CONTRACT_SPEC); launch/instruction/aggregate "
            "metrics are exercised by every admitted cell. Generated by "
            "tools/local_labels/emit_ac4_load_bearing_coverage.py from "
            "tools/local_labels/metric_registry.json + label_admitted.json + "
            "sm86_cohort_regen_ledger.json + "
            "workloads/simbench_bw/verify_kernel_contracts.py."
        ),
        "min_slug_disjoint_families": _MIN_SLUG_DISJOINT,
        "admitted_evaluable_slug_count": len(eval_slugs),
        "admitted_evaluable_slugs": eval_slugs,
        "contract_locked_factorial_slugs": contract_locked,
        "contract_locked_factorial_families": families_of(contract_locked),
        "imported_cohort_slugs_no_op_class_contract": imported_no_contract,
        "metric_count": len(metrics_rows),
        "metrics": metrics_rows,
    }


def emit(path: Path | None = None) -> Path:
    artifact = build_coverage(load_registry())
    out_path = path if path is not None else _ARTIFACT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return out_path


if __name__ == "__main__":
    written = emit()
    art = build_coverage(load_registry())
    by_status: dict[str, list[str]] = {}
    for m in art["metrics"]:
        by_status.setdefault(m["status"], []).append(
            f"{m['key']}={m['slug_disjoint_count']}"
        )
    print(f"wrote {written}")
    print(f"  admitted+evaluable slugs: {art['admitted_evaluable_slug_count']}")
    print(f"  contract-locked factorial: {art['contract_locked_factorial_slugs']}")
    for status in sorted(by_status):
        print(f"  {status}: {by_status[status]}")
