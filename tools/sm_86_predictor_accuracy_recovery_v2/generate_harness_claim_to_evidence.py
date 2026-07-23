#!/usr/bin/env python3
"""Regenerate the claim-to-evidence map + the readiness / typed-
unavailable budget accounting DETERMINISTICALLY from the committed
per-(cell, metric_key) evidence harness.

This is the AC consumer of the harness: it reads ONLY the harness
artifact + the metric registry and emits

  - ``claim_to_evidence_map.md`` -- regenerated from the harness's
    per-metric buckets / promotion gates / achieved-vs-target /
    baseline panel / ablation deltas (no hand authoring), and
  - ``harness_readiness_budget_sm_86.json`` -- the readiness verdict
    + the per-tier typed-unavailable budget accounting.

The budget tiers map the harness tiers onto the registry risk tiers
(cross-checked): T1 = replay counts (harness T1), T2 = cache
hit-rate / stream (harness L1 + L2 + T2), T3 = timing (harness
cycles + ns). The gates:

  - **T1**: ZERO silent gaps -- every exercised T1 row is scored or
    typed with a reason; a silent (untyped) gap FAILS CLOSED.
  - **T2**: typed-unavailable share vs the <= 10% budget, reported
    honestly MET/MISS (DEC-7 directional).
  - **T3**: an EXPLICIT diagnostic budget -- the typed rows are an
    enumerated diagnostic budget (no silent gaps), recorded.

Composition only: no upstream artifact is mutated; the map + budget
rebuild deterministically from the committed harness (byte-identical
except generated_at_commit / the embedded commit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "tools" / "local_labels")):
    if p not in sys.path:
        sys.path.insert(0, p)

ARCH = "sm_86"
HARNESS = (REPO_ROOT / "data/sm_86_predictor_accuracy_recovery_v2"
           / "per_metric_evidence_harness_sm_86.json")
REGISTRY = REPO_ROOT / "tools/local_labels/metric_registry.json"
DEFAULT_MAP = (REPO_ROOT / "data/sm_86_predictor_accuracy_recovery_v2"
               / "claim_to_evidence_map.md")
DEFAULT_BUDGET = (REPO_ROOT / "data/sm_86_predictor_accuracy_recovery_v2"
                  / "harness_readiness_budget_sm_86.json")
DEFAULT_SCHEMA = _HERE / "harness_readiness_budget.schema.json"

# harness tier -> AC budget tier (registry risk tier)
_TIER_MAP = {
    "T1": "T1", "L1": "T2", "L2": "T2", "T2": "T2",
    "cycles": "T3", "ns": "T3",
}
_TIER_BUDGET = {
    "T1": {"kind": "zero_silent_gaps", "budget": 0},
    "T2": {"kind": "max_typed_share", "budget": 0.10},
    "T3": {"kind": "explicit_diagnostic_budget", "budget": None},
}
T2_TYPED_SHARE_BUDGET = 0.10


class BudgetError(Exception):
    """A hard budget gate (T1 silent gap) is violated."""


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.strip()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_t1_keys(registry: dict) -> set:
    metrics = registry.get("metrics", registry)
    return {m["ncu_metric"] for m in metrics
            if m.get("risk_tier") == "T1"}


def compute_budget(harness: dict, registry: dict) -> list:
    """Per-AC-tier typed-unavailable accounting recomputed from the
    harness per_cell_metric rows. Raises BudgetError on a T1 silent
    gap (fail-closed)."""
    rows = harness["per_cell_metric"]
    # Cross-check: every harness T1 metric_key is a registry T1
    # (replay-count) metric, so the tier mapping is grounded.
    t1_registry = _registry_t1_keys(registry)
    t1_harness = {r["metric_key"] for r in rows if r["tier"] == "T1"}
    unknown = sorted(t1_harness - t1_registry)
    if unknown:
        raise BudgetError(
            f"harness T1 metric_keys not registry-T1 (replay_count): "
            f"{unknown}")

    by_ac: dict = {}
    for r in rows:
        if r["tier"] not in _TIER_MAP:
            raise BudgetError(
                f"unknown harness tier {r['tier']!r}; cannot map to "
                f"an AC budget tier")
        ac_tier = _TIER_MAP[r["tier"]]
        agg = by_ac.setdefault(ac_tier, {
            "n_total": 0, "n_scored": 0, "n_diagnostic": 0,
            "n_typed_unavailable": 0, "n_silent_gaps": 0,
            "harness_tiers": set()})
        agg["harness_tiers"].add(r["tier"])
        agg["n_total"] += 1
        b = r["bucket"]
        if b in ("main", "appendix"):
            agg["n_scored"] += 1
        elif b == "diagnostic":
            agg["n_diagnostic"] += 1
        else:  # typed_unavailable
            agg["n_typed_unavailable"] += 1
            if not r.get("typed_reason"):
                agg["n_silent_gaps"] += 1

    out = []
    for ac_tier in ("T1", "T2", "T3"):
        agg = by_ac[ac_tier]
        n_total = agg["n_total"]
        n_typed = agg["n_typed_unavailable"]
        share = n_typed / n_total if n_total else 0.0
        spec = _TIER_BUDGET[ac_tier]
        # A silent (untyped) typed-unavailable row is ALWAYS a
        # producer-integrity bug -- fail closed on EVERY tier (the
        # plan names T1=0 silent gaps explicitly; T2/T3 inherit the
        # same no-silent-gap baseline beneath their share / diagnostic
        # budgets).
        if agg["n_silent_gaps"] != 0:
            raise BudgetError(
                f"{ac_tier} budget violated: {agg['n_silent_gaps']} "
                f"silent (untyped) typed-unavailable gap(s); every "
                f"exercised row must be scored or typed with a "
                f"reason")
        if ac_tier == "T1":
            met = True
            budget_value = 0
        elif ac_tier == "T2":
            met = share <= T2_TYPED_SHARE_BUDGET
            budget_value = T2_TYPED_SHARE_BUDGET
        else:  # T3 explicit diagnostic budget
            # The typed rows are an enumerated diagnostic budget; with
            # silent gaps already rejected, the budget is satisfied.
            met = True
            budget_value = n_typed
        out.append({
            "tier": ac_tier,
            "budget_kind": spec["kind"],
            "budget": budget_value,
            "harness_tiers": sorted(agg["harness_tiers"]),
            "n_total": n_total,
            "n_scored": agg["n_scored"],
            "n_diagnostic": agg["n_diagnostic"],
            "n_typed_unavailable": n_typed,
            "n_silent_gaps": agg["n_silent_gaps"],
            "typed_unavailable_share": share,
            "met": met,
        })
    return out


def _metric_rows(harness: dict) -> list:
    """One readiness row per (tier, metric_key) from per_metric +
    bucket tallies from per_cell_metric."""
    cell_tally: dict = {}
    for r in harness["per_cell_metric"]:
        key = (r["tier"], r["metric_key"])
        cell_tally.setdefault(key, {"main": 0, "appendix": 0,
                                    "diagnostic": 0,
                                    "typed_unavailable": 0})
        cell_tally[key][r["bucket"]] += 1
    rows = []
    for m in harness["per_metric"]:
        key = (m["tier"], m["metric_key"])
        tally = cell_tally.get(key, {"main": 0, "appendix": 0,
                                     "diagnostic": 0,
                                     "typed_unavailable": 0})
        gates = m["gate"].get("promotion_gates", [])
        worst = None
        for g in gates:
            # Consume the harness's own authoritative per-gate verdict
            # (`met`) when present rather than re-deriving the
            # comparison direction; fall back to achieved>target only
            # when met is absent.
            met_flag = g.get("met")
            a, t = g.get("achieved"), g.get("target")
            failing = (met_flag is False) or (
                met_flag is None and a is not None and t is not None
                and a > t)
            if failing:
                worst = g
                break
        if m["promotion"] == "main":
            blocker = None
        elif m["n_scored"] == 0:
            blocker = "no scored cells (all rows typed-unavailable)"
        elif worst is not None:
            blocker = (f"gate {worst['gate']}: achieved "
                       f"{worst['achieved']} vs target "
                       f"{worst['target']} (MISS)")
        elif m["is_fitted"] and not m["gate"]["slug_disjoint_evidence"]:
            blocker = "missing slug-disjoint validation evidence"
        else:
            blocker = "non-vacuity floor or achieved-vs-target unmet"
        rows.append({
            "tier": m["tier"], "metric_key": m["metric_key"],
            "ac_tier": _TIER_MAP[m["tier"]],
            "promotion": m["promotion"],
            "is_fitted": m["is_fitted"],
            "n_scored": m["n_scored"],
            "n_scored_with_error": m["n_scored_with_error"],
            "n_diagnostic": m["n_diagnostic"],
            "n_typed_unavailable": m["n_typed_unavailable"],
            "bucket_tally": tally,
            # Carry the harness's full per-metric evidence VERBATIM so
            # the map/readiness faithfully reflect the source of truth
            # (a passing metric's achieved/target/CI are visible, and
            # a mutation to an achieved value changes this output).
            "achieved_vs_target": m["achieved_vs_target"],
            "ci": m["ci"],
            "strongest_allowed_claim": (
                "headline / replacement-grade (main bucket)"
                if m["promotion"] == "main"
                else "diagnostic / appendix scalar-subset result"),
            "promotion_blocker": blocker,
        })
    return sorted(rows, key=lambda r: (r["ac_tier"], r["tier"],
                                       r["metric_key"]))


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _fmt_ci(ci: dict) -> str:
    if ci.get("status") == "computed":
        return (f"{ci['statistic']} {_fmt(ci['mean'])} "
                f"[{_fmt(ci['lo'])}, {_fmt(ci['hi'])}] (n={ci['n']})")
    return f"ci_unavailable: {ci.get('reason', '?')} (n={ci.get('n')})"


def render_claim_map(harness: dict, budget: list,
                     metric_rows: list) -> str:
    """Deterministic markdown derived from the harness (no volatile
    timestamps; the source commit lives in the budget JSON)."""
    main_metrics = [r for r in metric_rows if r["promotion"] == "main"]
    overall = ("not_submission_grade"
               if any(not b["met"] for b in budget) or not main_metrics
               else "submission_grade_candidate")
    lines = [
        "# sm_86 Claim-To-Evidence Map (harness-derived)",
        "",
        f"Regenerated from `{HARNESS.relative_to(REPO_ROOT)}` "
        f"(config_digest `{harness['config_digest'][:12]}`).",
        "",
        f"Overall status: `{overall}`.",
        "",
        "| Tier | Metric | Promotion | Buckets (m/a/d/t) | "
        "Achieved-vs-target | CI | Strongest allowed claim | "
        "Promotion blocker |",
        "|------|--------|-----------|-------------------|----------"
        "----------|----|-------------------------|-----------------"
        "--|",
    ]
    for r in metric_rows:
        blocker = r["promotion_blocker"] or "-"
        t = r["bucket_tally"]
        buckets = (f"{t['main']}/{t['appendix']}/{t['diagnostic']}/"
                   f"{t['typed_unavailable']}")
        avt = " ; ".join(
            f"{g['gate']}={_fmt(g['achieved'])} vs "
            f"{_fmt(g['target'])} "
            f"({'MET' if g.get('met') else 'MISS'})"
            for g in r["achieved_vs_target"]) or "-"
        lines.append(
            f"| {r['ac_tier']}/{r['tier']} | `{r['metric_key']}` | "
            f"{r['promotion']} | {buckets} | {avt} | "
            f"{_fmt_ci(r['ci'])} | {r['strongest_allowed_claim']} | "
            f"{blocker} |")
    lines += ["", "## Typed-unavailable budget", "",
              "| Tier | Budget | Total | Scored | Typed | Silent gaps "
              "| Typed share | Met |",
              "|------|--------|-------|--------|-------|------------|"
              "-------------|-----|"]
    for b in budget:
        lines.append(
            f"| {b['tier']} | {b['budget_kind']} ({b['budget']}) | "
            f"{b['n_total']} | {b['n_scored']} | "
            f"{b['n_typed_unavailable']} | {b['n_silent_gaps']} | "
            f"{b['typed_unavailable_share']:.4f} | "
            f"{'MET' if b['met'] else 'MISS'} |")
    lines += [
        "", "## Baseline panel (sealed holdout, one governed touch)",
        "",
        f"Chosen model: `{harness['baseline_panel']['chosen_model']}` "
        f"on `{harness['baseline_panel']['metric_key']}`.", "",
        "| Baseline | Status | mean_abs_pp |",
        "|----------|--------|-------------|"]
    for e in harness["baseline_panel"]["entries"]:
        v = (f"{e['mean_abs_error_pp']:.3f}"
             if e["status"] == "computed" else "typed")
        lines.append(f"| {e['baseline']} | {e['status']} | {v} |")
    lines += ["", "## Component ablations (delta vs chosen >= 0)", "",
              "| Ablation | Metric | value_kind | delta_vs_chosen |",
              "|----------|--------|-----------|-----------------|"]
    for e in sorted(harness["ablations"]["entries"],
                    key=lambda e: e["ablation"]):
        d = e["delta_vs_chosen"]
        lines.append(
            f"| {e['ablation']} | `{e['metric_key']}` | "
            f"{e['value_kind']} | "
            f"{'n/a' if d is None else f'{d:+.4f}'} |")
    lines += [
        "", "## Rejected claim patterns", "",
        "- Any appendix-bucket metric described as publication-grade "
        "(its promotion gate MISSED).",
        "- Any tier whose typed-unavailable budget is MISS described "
        "as fully covered.",
        "- A scalar-only report that omits the typed-unavailable "
        "counts, the typed reasons, the baseline panel, or the "
        "ablation deltas.",
        "",
        "No metric may be described as publication-grade until its "
        "promotion gate passes AND its tier's typed-unavailable "
        "budget is MET.", ""]
    return "\n".join(lines)


def build_budget_artifact(harness: dict, registry: dict,
                          claim_map_md: str, budget: list,
                          metric_rows: list) -> dict:
    main_metrics = [r for r in metric_rows if r["promotion"] == "main"]
    overall = ("not_submission_grade"
               if any(not b["met"] for b in budget) or not main_metrics
               else "submission_grade_candidate")
    return {
        "schema_version": 1,
        "artifact_kind": "harness_readiness_budget",
        "arch": ARCH,
        "generated_at_commit": _git_head(),
        "harness_source": str(HARNESS.relative_to(REPO_ROOT)),
        "harness_config_digest": harness["config_digest"],
        "claim_map_path": str(DEFAULT_MAP.relative_to(REPO_ROOT)),
        "claim_map_sha256": hashlib.sha256(
            claim_map_md.encode("utf-8")).hexdigest(),
        "tier_budgets": budget,
        "readiness": {
            "overall_status": overall,
            "n_main_metrics": len(main_metrics),
            "per_metric": metric_rows,
        },
    }


def generate(*, map_out: Path = DEFAULT_MAP,
             budget_out: Path = DEFAULT_BUDGET,
             schema_path: Path = DEFAULT_SCHEMA) -> dict:
    harness = _load(HARNESS)
    registry = _load(REGISTRY)
    budget = compute_budget(harness, registry)
    metric_rows = _metric_rows(harness)
    claim_md = render_claim_map(harness, budget, metric_rows)
    artifact = build_budget_artifact(
        harness, registry, claim_md, budget, metric_rows)
    import jsonschema
    jsonschema.validate(artifact, _load(schema_path))
    map_out.write_text(claim_md, encoding="utf-8")
    with budget_out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-out", type=Path, default=DEFAULT_MAP)
    parser.add_argument(
        "--budget-out", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--schema-path", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)
    art = generate(map_out=args.map_out, budget_out=args.budget_out,
                   schema_path=args.schema_path)
    tb = {b["tier"]: ("MET" if b["met"] else "MISS")
          for b in art["tier_budgets"]}
    print(f"generate_harness_claim_to_evidence: wrote {args.map_out} "
          f"+ {args.budget_out} (status "
          f"{art['readiness']['overall_status']}; budgets {tb})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
