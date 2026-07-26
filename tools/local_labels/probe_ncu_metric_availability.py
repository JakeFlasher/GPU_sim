"""NCU metric-availability probe for the local sm_86 card.

Resolves, for every target metric named by ``tools/local_labels/read_ncu_report.py``
(its ``get_*`` helper functions) plus every metric in the committed
``metric_registry.json``, the exact rolled-up Nsight Compute metric id, its NCU
type and unit, and whether it is AVAILABLE on this device — by querying NCU's
own metric catalog (``ncu --query-metrics-mode suffix --metrics <bases> --csv``),
NOT by guessing. The probe is the only metric resolution authority: a metric
NCU does not list for this device is recorded ``available: false`` with a typed
``excluded_reason`` and is NEVER stubbed with a fabricated value, so a predictor
head that targets an unavailable/unresolved metric can be made to fail closed
against the committed probe.

The probe writes a committed artifact to
``data/local_labels/ncu_metric_availability_probe_sm_86.json`` in the
label-evidence namespace. Metric-catalog probing is an independently authorized
availability check and does not mint campaign labels; this query enumerates
metric names only — no kernel replay, no clock-control, no ``.ncu-rep`` artifact,
no privileged CUPTI.

Provenance of the artifact is ``(device, ncu_version)``: a metric-availability
catalog is a function of the card + the NCU release, reproducible by re-running
the query, so the artifact records those rather than a git source_commit.

Testability: ``query_ncu_metric_catalog`` accepts an injected ``csv_text`` so the
resolver can be exercised against real-shaped NCU CSV without a live device.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_DEFAULT_OUT = (
    _REPO_ROOT / "data" / "local_labels" / "ncu_metric_availability_probe_sm_86.json"
)

PROBE_SCHEMA_VERSION = 1

# Target metrics named by tools/local_labels/read_ncu_report.py's get_* helper
# functions, each mapped to its candidate canonical rolled-up NCU id. The probe
# confirms or rejects each candidate against NCU's catalog; ambiguous-rollup
# candidates that NCU does not list are recorded excluded (typed reason) for
# manual follow-up rather than silently resolved. get_L1_Total_Requests has no
# obvious 1:1 NCU name; its candidate is l1tex__t_requests.sum (the total
# T-stage request count), confirmed by the probe.
READ_NCU_REPORT_TARGETS: dict[str, str] = {
    "get_dram__sectors_read_sum": "dram__sectors_read.sum",
    "get_dram__sectors_write_sum": "dram__sectors_write.sum",
    "get_gpc__cycles_elapsed_avg_per_second": "gpc__cycles_elapsed.avg.per_second",
    "get_gpc__cycles_elapsed_max": "gpc__cycles_elapsed.max",
    "get_gpu__time_duration_sum": "gpu__time_duration.sum",
    "get_l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st_sum": "l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum",
    "get_l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld_sum": "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum",
    "get_l1tex__t_requests_pipe_lsu_mem_global_op_atom_sum": "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "get_l1tex__t_requests_pipe_lsu_mem_global_op_ld_sum": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "get_l1tex__t_requests_pipe_lsu_mem_global_op_red_sum": "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "get_l1tex__t_requests_pipe_lsu_mem_global_op_st_sum": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "get_l1tex__t_sector_hit_rate_pct": "l1tex__t_sector_hit_rate.pct",
    "get_l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate_pct": "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "get_l1tex__t_sectors_pipe_lsu_mem_global_op_ld_sum": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "get_l1tex__t_sectors_pipe_lsu_mem_global_op_st_sum": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "get_L1_Total_Requests": "l1tex__t_requests.sum",
    "get_launch__occupancy_limit_blocks": "launch__occupancy_limit_blocks",
    "get_launch__occupancy_limit_registers": "launch__occupancy_limit_registers",
    "get_launch__occupancy_limit_shared_mem": "launch__occupancy_limit_shared_mem",
    "get_launch__occupancy_limit_warps": "launch__occupancy_limit_warps",
    "get_lts__t_requests_srcunit_tex_sum": "lts__t_requests_srcunit_tex.sum",
    "get_lts__t_sector_hit_rate_pct": "lts__t_sector_hit_rate.pct",
    "get_sm__cycles_active_avg": "sm__cycles_active.avg",
    "get_sm__inst_executed_avg_per_cycle_elapsed": "sm__inst_executed.avg.per_cycle_elapsed",
    "get_sm__maximum_warps_avg_per_active_cycle": "sm__maximum_warps_avg_per_active_cycle",
    "get_sm__maximum_warps_per_active_cycle_pct": "sm__maximum_warps_per_active_cycle_pct",
    "get_smsp__inst_executed_sum": "smsp__inst_executed.sum",
    "get_sm__warps_active_avg_pct_of_peak_sustained_active": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "get_sm__warps_active_avg_per_cycle_active": "sm__warps_active.avg.per_cycle_active",
}


def _base_metric(metric: str) -> str:
    """Return the base metric (the portion before the first rollup ``.``).

    ``gpc__cycles_elapsed.max`` -> ``gpc__cycles_elapsed``;
    ``launch__occupancy_limit_warps`` -> itself (scalar metric, no rollup)."""
    return metric.split(".", 1)[0]


@dataclass(frozen=True)
class CatalogEntry:
    ncu_type: str
    ncu_unit: str
    available: bool


def query_ncu_metric_catalog(
    ncu_bin: str,
    bases: Iterable[str],
    *,
    csv_text: str | None = None,
) -> tuple[dict[str, CatalogEntry], str]:
    """Query NCU's catalog for every rolled-up form of ``bases`` and return
    ``(catalog, device_name)`` where ``catalog`` maps each rolled-up metric name
    to a :class:`CatalogEntry`.

    Runs ``ncu --query-metrics-mode suffix --metrics <comma-bases> --csv`` once.
    The CSV header is ``Metric name, Metric type, Metric unit, Metric
    description, <device-name>``; the final column is the per-device
    availability flag (``"1"`` available, ``"0"`` not). When ``csv_text`` is
    supplied the subprocess is skipped (used by tests to feed real-shaped CSV).
    """
    bases = list(dict.fromkeys(bases))
    if csv_text is None:
        cmd = [
            ncu_bin,
            "--query-metrics-mode",
            "suffix",
            "--metrics",
            ",".join(bases),
            "--csv",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{ncu_bin} --query-metrics exited {proc.returncode}; "
                f"stderr={proc.stderr.strip()!r}"
            )
        csv_text = proc.stdout
    reader = csv.reader(io.StringIO(csv_text))
    try:
        header = next(reader)
    except StopIteration:
        return {}, ""
    device_name = header[4] if len(header) >= 5 else ""
    catalog: dict[str, CatalogEntry] = {}
    for row in reader:
        if len(row) < 5:
            continue
        name, mtype, munit, _desc, avail_flag = row[0], row[1], row[2], row[3], row[4]
        catalog[name] = CatalogEntry(
            ncu_type=mtype,
            ncu_unit=munit,
            available=str(avail_flag).strip() == "1",
        )
    return catalog, device_name


def _target_list() -> list[dict]:
    """The union of read_ncu_report.py get_* targets and metric_registry.json
    metrics, deduplicated by canonical ncu_metric, with the source recorded."""
    from metric_registry import load_registry

    by_metric: dict[str, dict] = {}
    for get_fn, metric in READ_NCU_REPORT_TARGETS.items():
        by_metric.setdefault(
            metric,
            {"ncu_metric": metric, "sources": set(), "get_function": get_fn},
        )
        by_metric[metric]["sources"].add("read_ncu_report")
    for entry in load_registry().entries:
        rec = by_metric.setdefault(
            entry.ncu_metric,
            {"ncu_metric": entry.ncu_metric, "sources": set(), "get_function": None},
        )
        rec["sources"].add("registry")
        rec["registry_key"] = entry.key
        rec["registry_unit"] = entry.unit
    return [by_metric[m] for m in sorted(by_metric)]


def build_probe(
    ncu_bin: str,
    *,
    ncu_version: str,
    csv_text: str | None = None,
) -> dict:
    """Build the probe artifact dict. ``csv_text`` (test-injected) bypasses the
    live query; otherwise NCU is queried for every target's base metric."""
    targets = _target_list()
    bases = sorted({_base_metric(t["ncu_metric"]) for t in targets})
    catalog, device_name = query_ncu_metric_catalog(
        ncu_bin, bases, csv_text=csv_text
    )

    metrics_out: list[dict] = []
    for t in targets:
        metric = t["ncu_metric"]
        entry = catalog.get(metric)
        if entry is not None and entry.available:
            rec = {
                "ncu_metric": metric,
                "sources": sorted(t["sources"]),
                "available": True,
                "ncu_type": entry.ncu_type,
                "ncu_unit": entry.ncu_unit,
                "rollup": metric[len(_base_metric(metric)) + 1:] or "none",
            }
        else:
            # Typed exclusion — NOT stubbed with a fabricated value.
            if entry is not None:
                reason = "metric_listed_but_unavailable_on_device"
            elif _base_metric(metric).startswith("launch__"):
                # launch__* are launch-configuration attributes (theoretical
                # occupancy / static launch limits) computed from the launch
                # geometry and surfaced via NCU's Occupancy/LaunchStats
                # section; they are NOT hardware counters and so do not appear
                # in the --query-metrics counter catalog. This is a category
                # distinction, not a renamed/removed counter.
                reason = "launch_config_metric_not_in_counter_catalog"
            elif _base_metric(metric).startswith("sm__maximum_warps"):
                # sm__maximum_warps_avg_per_active_cycle (theoretical active
                # warps) and sm__maximum_warps_per_active_cycle_pct (theoretical
                # occupancy) are NCU occupancy-calculator special metrics, not
                # PerfWorks hardware counters — confirmed by Codex's task4
                # ratification — so they never appear in the --query-metrics
                # counter catalog. Theoretical occupancy stays a static-derivable
                # feature, never an NCU-counter label.
                reason = "occupancy_calculator_metric_not_in_counter_catalog"
            else:
                reason = "metric_not_in_ncu_catalog_on_device"
            rec = {
                "ncu_metric": metric,
                "sources": sorted(t["sources"]),
                "available": False,
                "excluded_reason": reason,
                "base_metric": _base_metric(metric),
            }
        if t.get("get_function"):
            rec["read_ncu_report_get_function"] = t["get_function"]
        if t.get("registry_key"):
            rec["registry_key"] = t["registry_key"]
        metrics_out.append(rec)

    available = {m["ncu_metric"] for m in metrics_out if m["available"]}
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "device": device_name,
        "arch": "sm_86",
        "ncu_metric_version": ncu_version,
        "query_command": "ncu --query-metrics-mode suffix --metrics <bases> --csv",
        "n_targets": len(metrics_out),
        "n_available": len(available),
        "n_excluded": len(metrics_out) - len(available),
        # Pin the two open resolution questions from the plan explicitly.
        "get_L1_Total_Requests_resolved_to": (
            "l1tex__t_requests.sum"
            if "l1tex__t_requests.sum" in available
            else None
        ),
        "gpu__time_duration_sum_confirmed": "gpu__time_duration.sum" in available,
        "metrics": metrics_out,
    }


class MetricNotProbeAvailableError(ValueError):
    """Raised when a predictor head / label target references a metric the
    committed probe did not confirm available on this device — the fail-closed
    mechanism that makes a head targeting an unprobed/unavailable metric fail
    rather than silently treat it as collectable."""


def load_probe(path: str | Path | None = None) -> dict:
    """Load the committed probe artifact (default the sm_86 artifact)."""
    p = Path(path) if path is not None else _DEFAULT_OUT
    return json.loads(p.read_text(encoding="utf-8"))


def available_metrics(probe: dict) -> frozenset[str]:
    """Set of ncu_metric ids the probe confirmed available on this device."""
    return frozenset(
        m["ncu_metric"] for m in probe.get("metrics", []) if m.get("available")
    )


def assert_target_metric_available(probe: dict, ncu_metric: str) -> None:
    """Fail closed if ``ncu_metric`` was not confirmed available by the probe.

    A head that targets a metric the probe marked excluded (or that was never
    probed) raises ``MetricNotProbeAvailableError`` — the unavailable metric is
    never silently treated as collectable."""
    by_metric = {m["ncu_metric"]: m for m in probe.get("metrics", [])}
    entry = by_metric.get(ncu_metric)
    if entry is None:
        raise MetricNotProbeAvailableError(
            f"metric {ncu_metric!r} is not in the metric-availability probe; "
            f"no head may target an unprobed metric"
        )
    if not entry.get("available"):
        raise MetricNotProbeAvailableError(
            f"metric {ncu_metric!r} is excluded by the probe "
            f"({entry.get('excluded_reason')!r}); no head may target it"
        )


def _normalize_ncu_version(ncu_bin: str) -> str:
    """Parse ``ncu --version`` and fold the 2026.1.<patch>(.<build>) banner to
    the pinned ``ncu_2026.1.x`` family form (mirrors the aggregator)."""
    try:
        proc = subprocess.run(
            [ncu_bin, "--version"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    import re

    for line in proc.stdout.splitlines():
        if re.match(r"^Version\s+2026\.1\.\d+", line):
            return "ncu_2026.1.x"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Query NCU metric availability for every target metric and write a "
            "committed probe artifact under data/local_labels/."
        )
    )
    parser.add_argument("--ncu", default="ncu", help="Path to the NCU CLI.")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args(argv)

    ncu_version = _normalize_ncu_version(args.ncu)
    try:
        probe = build_probe(args.ncu, ncu_version=ncu_version)
    except RuntimeError as exc:
        sys.stderr.write(f"probe_ncu_metric_availability: {exc}\n")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(probe, fh, indent=2, sort_keys=True)
        fh.write("\n")
    sys.stderr.write(
        f"probe_ncu_metric_availability: wrote {args.out} "
        f"({probe['n_available']}/{probe['n_targets']} available)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
