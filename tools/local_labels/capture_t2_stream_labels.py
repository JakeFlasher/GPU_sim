#!/usr/bin/env python3
"""Capture the T2 cache-stream traffic labels and land them on admitted
cells — the count-label mechanics REUSED VERBATIM over the registry's
cache_stream class.

The capture target set is the registry's cache_stream rows
(``dram__sectors_read.sum``, ``dram__sectors_write.sum``,
``lts__t_requests_srcunit_tex.sum``,
``l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum``,
``l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum``) — every one a
unit-tagged integer count confirmed available by the committed
metric-availability probe. The capture, gating, integer-stability
admission, and additive in-place splice are the EXISTING
``capture_count_labels`` machinery parameterized by this metric set:
the rule has no tunable parameters, repeats must agree exactly (a
jittering DRAM-level counter is typed ``label_count_unstable`` with
its raw repeat values recorded, never averaged), and untouched store
fields stay byte-identical.

The 2 probe-EXCLUDED DRAM op-split aliases
(``dram__sectors_op_read.sum``, ``dram__sectors_op_write.sum``) are
NEVER captured: the probe gate
(``assert_targets_probe_available``) fail-closes any capture set
carrying them, and this driver's exclusions artifact records them as
standing typed fail-closed rows with the committed probe evidence —
resurrecting an alias requires fresh probe evidence by construction.

CLAUDE.md compliance (binding): identical to the count capture — NCU-derived
labels remain targets rather than simulator features, and every output lands
under ``data/local_labels/`` (QuarantinePathError fail-closed); CSV-stdout
capture, no ``.ncu-rep``; no driver-root operations; no privileged CUPTI;
sm_86-only with sha256-pinned binary binding.

CLI:
  python tools/local_labels/capture_t2_stream_labels.py \
    [--label-admitted data/local_labels/label_admitted.json] \
    [--ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu] \
    [--repeats 5] [--arch sm_86] [--dry-run]

  Exit 3 (no write) when ncu is not on the host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_LOCAL_LABELS = REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS))

from capture_count_labels import (  # noqa: E402
    DEFAULT_LABEL_ADMITTED,
    DEFAULT_NCU,
    DEFAULT_REPEATS,
    assert_quarantine_path,
    assert_targets_probe_available,
    build_governance_artifact,
    capture_and_land,
)

_ARCH = "sm_86"
DEFAULT_T2_EXCLUSIONS_OUT = (
    "data/local_labels/t2_stream_label_exclusions_sm_86.json"
)
DEFAULT_T2_GOVERNANCE_OUT = (
    "data/local_labels/t2_stream_label_capture_governance_sm_86.json"
)
DEFAULT_T2_AUDIT_OUT = (
    "data/local_labels/t2_stream_label_capture_audit_sm_86.json"
)


def t2_stream_metric_identifiers() -> tuple[str, ...]:
    """Every registry cache_stream unit-tagged count metric available
    on sm_86 — derived from the registry so the registry/probe
    lockstep governs what gets captured (never a hand-typed list)."""
    from metric_registry import load_registry

    out = []
    for e in load_registry().entries:
        if (
            e.metric_class == "cache_stream"
            and e.carrier == "unit_tagged_metric"
            and e.unit == "count"
            and e.availability == "available"
            and (e.per_arch_availability or {}).get("sm_86")
            == "available"
        ):
            out.append(e.ncu_metric)
    return tuple(sorted(out))


def excluded_dram_alias_rows() -> list[dict[str, Any]]:
    """The registry's probe-excluded DRAM op-split alias rows as
    standing typed fail-closed exclusion records, carrying the
    committed probe evidence verbatim (the negative arm's positive
    counterpart: the aliases stay typed unless the probe itself
    re-admits them)."""
    from metric_registry import load_registry_raw
    from probe_ncu_metric_availability import load_probe

    raw = load_registry_raw()
    rows = raw["metrics"] if isinstance(raw, dict) and "metrics" in raw \
        else raw
    aliases = sorted(
        r["ncu_metric"]
        for r in (rows if isinstance(rows, list) else rows.values())
        if r.get("availability") == "excluded"
        and str(r.get("ncu_metric", "")).startswith("dram__sectors_op_")
    )
    probe = load_probe()
    probe_by_metric = {
        t.get("ncu_metric"): t for t in probe.get("metrics", [])
    }
    out: list[dict[str, Any]] = []
    for ncu_metric in aliases:
        evidence = probe_by_metric.get(ncu_metric)
        if evidence is None or evidence.get("available") is not False:
            raise RuntimeError(
                f"alias {ncu_metric} lacks committed probe-excluded "
                "evidence; refusing to emit a typed row without it"
            )
        out.append({
            "ncu_metric": ncu_metric,
            "typed_reason": evidence.get(
                "excluded_reason",
                "metric_not_in_ncu_catalog_on_device",
            ),
            "detail": (
                "probe-excluded DRAM op-split alias of the captured "
                "dram__sectors_read/write.sum counters; stays a "
                "typed fail-closed row — the capture probe gate "
                "rejects any target set carrying it, so "
                "resurrection requires fresh committed probe "
                "evidence"
            ),
            "probe_evidence": {
                "probe_artifact": (
                    "data/local_labels/"
                    "ncu_metric_availability_probe_sm_86.json"
                ),
                "available": evidence.get("available"),
                "excluded_reason": evidence.get("excluded_reason"),
                "registry_key": evidence.get("registry_key"),
            },
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-admitted", default=DEFAULT_LABEL_ADMITTED)
    parser.add_argument("--ncu", default=DEFAULT_NCU)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--arch", default=_ARCH)
    parser.add_argument(
        "--exclusions-out", default=DEFAULT_T2_EXCLUSIONS_OUT)
    parser.add_argument(
        "--governance-out", default=DEFAULT_T2_GOVERNANCE_OUT)
    parser.add_argument("--audit-out", default=DEFAULT_T2_AUDIT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    label_path = Path(args.label_admitted)
    exclusions_path = Path(args.exclusions_out)
    governance_path = Path(args.governance_out)
    audit_path = Path(args.audit_out)
    for p in (label_path, exclusions_path, governance_path, audit_path):
        assert_quarantine_path(p)

    metrics = t2_stream_metric_identifiers()
    # Fail closed BEFORE any capture if the registry-derived set ever
    # drifts to include an unprobed/excluded identifier (the alias
    # resurrection guard).
    assert_targets_probe_available(metrics)

    if args.dry_run:
        sys.stderr.write(
            "[capture_t2_stream] --dry-run: no files written\n")
        return 0

    try:
        admitted, exclusions, stats = capture_and_land(
            label_admitted_path=label_path,
            ncu=args.ncu,
            repeats=args.repeats,
            arch=args.arch,
            metrics=metrics,
        )
    except FileNotFoundError as exc:
        sys.stderr.write(
            f"t2_stream_capture_unavailable: {exc} (no write performed)\n"
        )
        return 3

    label_path.write_text(
        json.dumps(admitted, indent=2) + "\n", encoding="utf-8",
    )
    excl_artifact = {
        "schema_version": 1,
        "artifact_kind": "t2_stream_label_exclusions",
        "arch": args.arch,
        "captured_metrics": list(metrics),
        "excluded_alias_rows": excluded_dram_alias_rows(),
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
    governance["artifact_kind"] = "t2_stream_label_capture_governance"
    governance["metric_class"] = "cache_stream"
    governance_path.write_text(
        json.dumps(governance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_cells = stats.get("audit_cells", [])
    audit_artifact = {
        "schema_version": 1,
        "artifact_kind": "t2_stream_label_capture_audit",
        "arch": args.arch,
        "n_repeats": args.repeats,
        "n_cells": len(audit_cells),
        "note": (
            "Per-cell per-metric raw repeat values + integer-stability "
            "statistics for the T2 cache-stream capture. Every "
            "admitted T2 label must show all_integral=true and "
            "integer_spread=0.0 over n_repeats repeats; "
            "admitted=false rows are the typed per-metric exclusions "
            "(DRAM-level counters that jitter across repeats land "
            "here with their raw values, never averaged)."
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
