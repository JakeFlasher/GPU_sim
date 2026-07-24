"""Emit the sm_86 expanded NCU read-only-sector label ledger.

Two phases, decoupled so the ledger emission is deterministic:

1. ``harvest`` — for every classifiable DONE cell that has a probe binary
   under ``calibration_cohort_b/``, run NCU 3 times on the probe binary
   (pinned read-only-sector metric family, ``--clock-control none``,
   ``--launch-skip 5`` — the probes loop 6x so the 6th, fully-warmed launch
   is measured), parse each run's CSV into the six raw L1/L2
   read-only fields, compute per-metric mean + relative CV, quarantine
   CV > 0.05 with the typed ``ncu_repeatability_failed`` reason, and write
   one ``<slug>.json`` sidecar (raw counters + rates + argv provenance) per
   cell into ``ncu_expanded_sidecars_sm86/``.
2. ``emit`` — read the sidecars on disk + the frozen 5-cell seed ledger,
   recompute every sidecar SHA, and write
   ``label_ledger_v1_sm86_expanded.toml`` covering the full classifiable
   set: the 5 seed rows preserved verbatim (one-way cross-stamp; the seed
   bytes are never edited), the harvested cohort_b rows, and a typed
   ``probe_binary_missing_for_slug`` row for every classifiable DONE cell
   that has no probe binary yet (next-round build-out, NOT a silent skip).

The ``emit`` phase reads only the on-disk sidecars + seed, so re-running it
reproduces the ledger byte-for-byte.

CLAUDE.md Rule 1 (NCU calibration-only) and Rule 2 (no clock-policy
mutation) hold: see the package docstring and
``ncu_harvester._build_invocation``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[4]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import statistics
from datetime import datetime, timezone
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[5]

from sol_baselines_harness.calibration.cache_prediction.ncu_harvester import (  # noqa: E402
    NCU_REQUIRED_METRICS,
    NcuHarvesterError,
    _build_invocation,
    _run_checked_ncu,
)
from sol_baselines_harness.calibration.cache_prediction.label_parser import (  # noqa: E402
    NcuLabelParserError,
    parse_ncu_csv,
)

_CAL_ROOT: Final[Path] = (
    _REPO_ROOT / "scripts" / "baselines_harness" / "calibration"
)
_CACHE_PRED_ROOT: Final[Path] = _CAL_ROOT / "cache_prediction"
_COVERAGE_LEDGER: Final[Path] = _CAL_ROOT / "sm86_coverage_ledger_v1.toml"
_SEED_LEDGER: Final[Path] = _CACHE_PRED_ROOT / "label_ledger_v1_sm86_pilot_v1.toml"
_COHORT_B_ROOT: Final[Path] = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "tests" / "calibration_cohort_b"
)
_SIDECAR_DIR: Final[Path] = _CACHE_PRED_ROOT / "ncu_expanded_sidecars_sm86"
_EXPANDED_LEDGER: Final[Path] = _CACHE_PRED_ROOT / "label_ledger_v1_sm86_expanded.toml"

# Path to a non-PATH NCU install on this calibration host. _build_invocation
# emits a bare ``ncu`` argv and _run_checked_ncu resolves it via $PATH, so the
# harvest phase prepends the nsight-compute dir to $PATH before invoking.
_NCU_SEARCH_DIRS: Final[tuple[str, ...]] = (
    "/usr/local/cuda-13.2/nsight-compute-2026.1.0",
)

SIDECAR_SCHEMA_VERSION: Final[int] = 2
LEDGER_SCHEMA_VERSION: Final[int] = 1
HOST_ARCH: Final[str] = "sm_86"
HOST_DESCRIPTOR_HASH: Final[str] = "local_sm_86_3090_ti_24gib"
CV_CAP: Final[float] = 0.05
# The cohort_b probes loop their kernel 6 times by design so NCU
# `--launch-skip 5 --launch-count 1` measures the 6th, fully-warmed launch
# (the Makefile-documented warm-steady-state protocol).
LAUNCH_SKIP_COUNT: Final[int] = 5
NCU_RUNS_PER_CELL: Final[int] = 3

# Closed reason vocabulary for a non-completed harvested row. The first
# two are genuine HARVEST failures (a complete ledger must carry zero of
# these); the last two are legitimate TERMINAL outcomes a complete ledger
# may carry — a CV quarantine (jitter above cap, with
# three parsed runs) and a zero-read-only-traffic kernel (NCU's read-only
# sector hit rate is undefined for write-only / index-generated kernels;
# the framework's canonical terminal — accuracy_diagnostic.detect_zero_
# read_unavailable_row). `probe_binary_missing_for_slug` lives on the
# emit-side probe_pending rows, not on a harvested sidecar.
HARVEST_FAILURE_REASONS: Final[tuple[str, ...]] = (
    "ncu_subprocess_failed",
    "ncu_metric_missing",
)
HARVEST_TERMINAL_REASONS: Final[tuple[str, ...]] = (
    "ncu_repeatability_failed",
    "ncu_label_no_read_only_traffic",
)
FAILURE_REASON_VOCABULARY: Final[tuple[str, ...]] = (
    *HARVEST_FAILURE_REASONS,
    *HARVEST_TERMINAL_REASONS,
    "probe_binary_missing_for_slug",
)

# NCU read-only sector denominators (the per-level `..._op_read.sum`
# counters). When BOTH are zero the kernel issued no read-only traffic
# and the read-only hit rate is undefined.
_L1_READ_DENOM: Final[str] = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
_L2_READ_DENOM: Final[str] = "lts__t_sectors_srcunit_tex_op_read.sum"

_REQUIRED_RUN_FIELDS: Final[tuple[str, ...]] = (
    "l1_global_load_hit_rate",
    "l1_global_load_sectors_total",
    "l1_global_load_sectors_hit",
    "l2_read_hit_rate",
    "l2_read_sectors_total",
    "l2_read_sectors_hit",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ensure_ncu_on_path() -> None:
    if shutil.which("ncu") is not None:
        return
    for d in _NCU_SEARCH_DIRS:
        if (Path(d) / "ncu").is_file():
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return


def _relative_cv(samples: tuple[float, ...]) -> float:
    """Sample-stdev / mean. Uniformly-zero samples are trivially
    repeatable (CV 0.0). Mirrors ncu_repeatability._relative_cv."""
    if len(samples) < 2:
        return 0.0
    mean = statistics.fmean(samples)
    if mean == 0.0:
        return 0.0 if all(s == 0.0 for s in samples) else -1.0
    return statistics.stdev(samples) / mean


def discover_cohort_b_probes(repo_root: Path) -> dict[str, Path]:
    """Map slug -> probe binary for every executable (extension-less)
    file in calibration_cohort_b/."""
    root = repo_root / "nvbit_tracer" / "metric_extractor" / "tests" / "calibration_cohort_b"
    probes: dict[str, Path] = {}
    if not root.is_dir():
        return probes
    for entry in sorted(root.iterdir()):
        if entry.is_file() and os.access(entry, os.X_OK) and "." not in entry.name:
            probes[entry.name] = entry
    return probes


def load_done_slugs(repo_root: Path) -> set[str]:
    import tomllib
    ledger = repo_root / "scripts" / "baselines_harness" / "calibration" / "sm86_coverage_ledger_v1.toml"
    data = tomllib.loads(ledger.read_text())
    rows = data.get("cells") or data.get("rows") or []
    return {r["slug"] for r in rows if r.get("done") is True}


def build_slug_bucket_map(repo_root: Path) -> dict[str, str]:
    """slug -> predictor_bucket from the accuracy diagnostic's per-cell
    rows (the canonical problem->cohort classification authority)."""
    from sol_baselines_harness.calibration.cache_prediction.accuracy_diagnostic import (
        build_accuracy_diagnostic_report,
    )
    report = build_accuracy_diagnostic_report(repo_root=repo_root)
    return {row.problem_slug: row.predictor_bucket for row in report.per_cell_rows}


def _is_classifiable(bucket: str) -> bool:
    return bucket in ("predictor_friendly", "predictor_hostile")


_ZERO_READ_RUN: Final[dict] = {
    "l1_global_load_hit_rate": 0.0,
    "l1_global_load_sectors_total": 0,
    "l1_global_load_sectors_hit": 0,
    "l2_read_hit_rate": 0.0,
    "l2_read_sectors_total": 0,
    "l2_read_sectors_hit": 0,
}


def _extract_metric_values(csv_text: str) -> dict[str, float]:
    """Pull the pinned read-only metric values out of an NCU `--csv`
    stdout (first occurrence per metric; thousands separators stripped).
    Used to detect zero read-only traffic BEFORE `parse_ncu_csv`, which
    raises `ncu_label_denominator_zero` on a write-only kernel."""
    out: dict[str, float] = {}
    reader = csv.reader(io.StringIO(csv_text))
    name_idx = val_idx = None
    for row in reader:
        if name_idx is None:
            if "Metric Name" in row and "Metric Value" in row:
                name_idx = row.index("Metric Name")
                val_idx = row.index("Metric Value")
            continue
        if len(row) <= max(name_idx, val_idx):
            continue
        name = row[name_idx]
        if name in NCU_REQUIRED_METRICS and name not in out:
            try:
                out[name] = float(row[val_idx].replace(",", "").strip())
            except ValueError:
                pass
    return out


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_expanded.emit_expanded_ncu_ledger.harvest_cell')
def harvest_cell(slug: str, binary: Path) -> dict:
    """Run NCU 3x on the probe binary; return a sidecar dict (written by
    the caller). Never raises — failures / terminal outcomes surface as a
    typed reason. A kernel that issues no read-only traffic (both per-level
    `..._op_read.sum` denominators zero on every run) is the canonical
    terminal `ncu_label_no_read_only_traffic`, NOT a subprocess failure and
    NOT a fabricated read."""
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_expanded.emit_expanded_ncu_ledger.harvest_cell')
    binary_sha = _sha256_file(binary)
    argv = _build_invocation(
        binary=str(binary),
        launch_skip_count=LAUNCH_SKIP_COUNT,
    )
    runs: list[dict] = []
    argv_provenance: list[list[str]] = []
    failure_reason = ""
    zero_read_runs = 0
    for _ in range(NCU_RUNS_PER_CELL):
        argv_provenance.append(list(argv))
        try:
            completed = _run_checked_ncu(argv)
        except NcuHarvesterError:
            failure_reason = "ncu_subprocess_failed"
            break
        if not completed.stdout.strip():
            failure_reason = "ncu_subprocess_failed"
            break
        vals = _extract_metric_values(completed.stdout)
        if vals.get(_L1_READ_DENOM, -1.0) == 0.0 and vals.get(_L2_READ_DENOM, -1.0) == 0.0:
            runs.append(dict(_ZERO_READ_RUN))
            zero_read_runs += 1
            continue
        try:
            record = parse_ncu_csv(
                completed.stdout,
                ncu_metric_version="2026.1.0.0",
                binary_sha256=binary_sha,
                launch_skip_count=LAUNCH_SKIP_COUNT,
                warmup_protocol="launch_skip_5",
            )
        except NcuLabelParserError as exc:
            typed = getattr(exc, "typed_reason", "")
            failure_reason = (
                "ncu_metric_missing"
                if typed == "ncu_label_metric_missing"
                else "ncu_subprocess_failed"
            )
            break
        runs.append({
            "l1_global_load_hit_rate": float(record.l1_global_load_hit_rate),
            "l1_global_load_sectors_total": int(record.l1_global_load_sectors_total),
            "l1_global_load_sectors_hit": int(record.l1_global_load_sectors_hit),
            "l2_read_hit_rate": float(record.l2_read_hit_rate),
            "l2_read_sectors_total": int(record.l2_read_sectors_total),
            "l2_read_sectors_hit": int(record.l2_read_sectors_hit),
        })

    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sidecar: dict = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "slug": slug,
        "language": "cuda_cpp",
        "ncu_metric_family": list(NCU_REQUIRED_METRICS),
        "cv_cap": CV_CAP,
        "probe_binary_sha256": binary_sha,
        "launch_skip_count": LAUNCH_SKIP_COUNT,
        "argv_provenance": argv_provenance,
        "argv_clock_policy_flag": "",
        "captured_at_utc": captured_at,
    }
    # A genuine harvest failure (subprocess / metric) or a mix of zero-read
    # and read runs (inconsistent for a deterministic probe) is a failure.
    if failure_reason or len(runs) != NCU_RUNS_PER_CELL or (
        0 < zero_read_runs < NCU_RUNS_PER_CELL
    ):
        sidecar.update({
            "completed": False,
            "failure_reason": failure_reason or "ncu_subprocess_failed",
            "cv_quarantined": False,
            "no_read_only_traffic": False,
            "runs": runs,
            "relative_cv_l1": -1.0,
            "relative_cv_l2": -1.0,
            "ncu_l1_hit_rate_mean": -1.0,
            "ncu_l2_hit_rate_mean": -1.0,
        })
        return sidecar

    # Zero read-only traffic on all three runs: terminal, not a failure.
    if zero_read_runs == NCU_RUNS_PER_CELL:
        sidecar.update({
            "completed": False,
            "failure_reason": "ncu_label_no_read_only_traffic",
            "cv_quarantined": False,
            "no_read_only_traffic": True,
            "runs": runs,
            "relative_cv_l1": 0.0,
            "relative_cv_l2": 0.0,
            "ncu_l1_hit_rate_mean": 0.0,
            "ncu_l2_hit_rate_mean": 0.0,
        })
        return sidecar

    l1 = tuple(r["l1_global_load_hit_rate"] for r in runs)
    l2 = tuple(r["l2_read_hit_rate"] for r in runs)
    cv_l1 = _relative_cv(l1)
    cv_l2 = _relative_cv(l2)
    quarantined = (cv_l1 < 0.0) or (cv_l2 < 0.0) or (cv_l1 > CV_CAP) or (cv_l2 > CV_CAP)
    sidecar.update({
        "completed": not quarantined,
        "failure_reason": "ncu_repeatability_failed" if quarantined else "",
        "cv_quarantined": quarantined,
        "no_read_only_traffic": False,
        "runs": runs,
        "relative_cv_l1": cv_l1,
        "relative_cv_l2": cv_l2,
        "ncu_l1_hit_rate_mean": statistics.fmean(l1),
        "ncu_l2_hit_rate_mean": statistics.fmean(l2),
    })
    return sidecar


def _sidecar_is_fresh(path: Path) -> bool:
    """A sidecar already harvested under the current launch-skip protocol can
    be reused (lets a re-run RESUME if the NCU sweep died mid-cohort)."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return data.get("launch_skip_count") == LAUNCH_SKIP_COUNT


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_expanded.emit_expanded_ncu_ledger.run_harvest')
def run_harvest(repo_root: Path = _REPO_ROOT, *, resume: bool = True) -> dict:
    """Phase 1: harvest every classifiable DONE cell with a probe binary;
    write one sidecar per cell. Returns a summary dict. With ``resume`` (the
    default) a cell whose sidecar is already fresh for the current launch-skip
    protocol is skipped, so a re-run continues a cohort the NCU sweep left
    unfinished rather than re-profiling from scratch."""
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_expanded.emit_expanded_ncu_ledger.run_harvest')
    _ensure_ncu_on_path()
    probes = discover_cohort_b_probes(repo_root)
    done = load_done_slugs(repo_root)
    buckets = build_slug_bucket_map(repo_root)
    sidecar_dir = repo_root / _SIDECAR_DIR.relative_to(_REPO_ROOT)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    harvest_slugs = sorted(
        s for s in probes
        if s in done and _is_classifiable(buckets.get(s, ""))
    )
    summary = {"harvested": [], "quarantined": [], "failed": [], "skipped": []}
    for slug in harvest_slugs:
        out = sidecar_dir / f"{slug}.json"
        if resume and _sidecar_is_fresh(out):
            summary["skipped"].append(slug)
            print(f"  {slug}: skipped (fresh sidecar present)", flush=True)
            continue
        sidecar = harvest_cell(slug, probes[slug])
        sidecar["predictor_bucket"] = buckets.get(slug, "")
        (sidecar_dir / f"{slug}.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
        )
        if sidecar["completed"]:
            summary["harvested"].append(slug)
        elif sidecar["cv_quarantined"]:
            summary["quarantined"].append(slug)
        else:
            summary["failed"].append(slug)
        print(
            f"  {slug}: completed={sidecar['completed']} "
            f"reason={sidecar['failure_reason']!r} "
            f"l1={sidecar.get('ncu_l1_hit_rate_mean'):.4f} "
            f"l2={sidecar.get('ncu_l2_hit_rate_mean'):.4f} "
            f"cv_l1={sidecar.get('relative_cv_l1'):.4f} cv_l2={sidecar.get('relative_cv_l2'):.4f}",
            flush=True,
        )
    print(
        f"harvest summary: {len(summary['harvested'])} completed, "
        f"{len(summary['quarantined'])} quarantined, {len(summary['failed'])} failed "
        f"of {len(harvest_slugs)} probe-backed classifiable DONE cells",
        flush=True,
    )
    return summary


def _toml_s(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_expanded.emit_expanded_ncu_ledger.emit_ledger')
def emit_ledger(repo_root: Path = _REPO_ROOT) -> Path:
    """Phase 2: read the on-disk sidecars + frozen seed, write the expanded
    ledger deterministically (one-way seed cross-stamp; full classifiable
    set with probe_binary_missing_for_slug rows for unharvested cells)."""
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_expanded.emit_expanded_ncu_ledger.emit_ledger')
    import tomllib
    sidecar_dir = repo_root / _SIDECAR_DIR.relative_to(_REPO_ROOT)
    seed_path = repo_root / _SEED_LEDGER.relative_to(_REPO_ROOT)
    out_path = repo_root / _EXPANDED_LEDGER.relative_to(_REPO_ROOT)

    seed = tomllib.loads(seed_path.read_text())
    seed_sha = _sha256_file(seed_path)
    seed_rows = sorted(seed["entries"], key=lambda e: e["slug"])
    seed_slugs = {e["slug"] for e in seed_rows}

    done = load_done_slugs(repo_root)
    buckets = build_slug_bucket_map(repo_root)
    classifiable_done = sorted(
        s for s in done if _is_classifiable(buckets.get(s, "")) and s not in seed_slugs
    )

    sidecars: dict[str, dict] = {}
    for p in sorted(sidecar_dir.glob("*.json")) if sidecar_dir.is_dir() else []:
        sidecars[p.stem] = {"data": json.loads(p.read_text()), "sha256": _sha256_file(p),
                            "rel": str(p.relative_to(repo_root))}

    completed_rows: list[dict] = []
    cv_quarantined_rows: list[dict] = []
    no_read_rows: list[dict] = []
    failed_rows: list[dict] = []
    missing_rows: list[dict] = []
    for slug in classifiable_done:
        if slug in sidecars:
            sc = sidecars[slug]["data"]
            row = {
                "source": "cohort_b_probe", "slug": slug, "language": "cuda_cpp",
                "predictor_bucket": buckets.get(slug, ""),
                "completed": bool(sc["completed"]),
                "failure_reason": sc["failure_reason"],
                "ncu_l1_hit_rate_mean": sc["ncu_l1_hit_rate_mean"],
                "ncu_l2_hit_rate_mean": sc["ncu_l2_hit_rate_mean"],
                "ncu_l1_cv": sc["relative_cv_l1"], "ncu_l2_cv": sc["relative_cv_l2"],
                "ncu_run_count": len(sc["runs"]),
                "sidecar_path": sidecars[slug]["rel"],
                "sidecar_sha256": sidecars[slug]["sha256"],
                "probe_binary_sha256": sc["probe_binary_sha256"],
            }
            if sc["completed"]:
                completed_rows.append(row)
            elif sc["failure_reason"] == "ncu_repeatability_failed":
                cv_quarantined_rows.append(row)
            elif sc["failure_reason"] == "ncu_label_no_read_only_traffic":
                no_read_rows.append(row)
            else:
                failed_rows.append(row)
        else:
            missing_rows.append({
                "source": "probe_pending", "slug": slug, "language": "cuda_cpp",
                "predictor_bucket": buckets.get(slug, ""),
                "completed": False, "failure_reason": "probe_binary_missing_for_slug",
            })

    captured_at = max(
        (sidecars[s]["data"]["captured_at_utc"] for s in sidecars),
        default=seed["meta"].get("captured_at_utc", "1970-01-01T00:00:00+00:00"),
    )

    L: list[str] = []
    L.append("# sm_86 EXPANDED NCU read-only-sector label ledger.")
    L.append("# GENERATED (deterministic) by")
    L.append("# calibration/cache_prediction/sm86_expanded/emit_expanded_ncu_ledger.py")
    L.append("# from the on-disk ncu_expanded_sidecars_sm86/<slug>.json 3-run sidecars")
    L.append("# plus the frozen 5-cell seed ledger. ONE-WAY cross-stamp: the seed rows")
    L.append("# are preserved verbatim and the seed file's SHA is recorded in [meta];")
    L.append("# the seed (label_ledger_v1_sm86_pilot_v1.toml) is NEVER edited in place.")
    L.append("#")
    L.append("# Read-only-sector metric family only (CLAUDE.md Rule 1 carve-out); every")
    L.append("# NCU argv pins --clock-control none with no lock-clock flag (Rule 2).")
    L.append("")
    L.append("[meta]")
    L.append(f"schema_version = {LEDGER_SCHEMA_VERSION}")
    L.append(f"host_arch = {_toml_s(HOST_ARCH)}")
    L.append(f"host_descriptor_hash = {_toml_s(HOST_DESCRIPTOR_HASH)}")
    L.append(f"captured_at_utc = {_toml_s(captured_at)}")
    L.append(f"cv_cap = {CV_CAP!r}")
    L.append("ncu_metric_family = [" + ", ".join(_toml_s(m) for m in NCU_REQUIRED_METRICS) + "]")
    L.append(f"seed_ledger_path = {_toml_s(str(_SEED_LEDGER.relative_to(_REPO_ROOT)))}")
    L.append(f"seed_ledger_sha256 = {_toml_s(seed_sha)}")
    L.append(f"seed_admitted_count = {len(seed_rows)}")
    L.append(f"harvested_completed_count = {len(completed_rows)}")
    L.append(f"harvested_cv_quarantined_count = {len(cv_quarantined_rows)}")
    L.append(f"harvested_no_read_only_traffic_count = {len(no_read_rows)}")
    L.append(f"harvested_failed_count = {len(failed_rows)}")
    L.append(f"probe_pending_count = {len(missing_rows)}")
    L.append(
        f"row_count = "
        f"{len(seed_rows) + len(completed_rows) + len(cv_quarantined_rows) + len(no_read_rows) + len(failed_rows) + len(missing_rows)}"
    )
    L.append("")

    # Seed rows preserved verbatim (one-way cross-stamp).
    for e in seed_rows:
        L.append("[[entries]]")
        L.append('source = "seed"')
        L.append(f"slug = {_toml_s(e['slug'])}")
        L.append(f"language = {_toml_s(e.get('language', 'cuda_cpp'))}")
        L.append(f"predictor_bucket = {_toml_s(e.get('predictor_bucket', ''))}")
        L.append("completed = true")
        L.append('failure_reason = ""')
        # One-way cross-stamp: copy every cross-stamp SHA the seed row records
        # (its keys end in `_sha256`) under a `seed_` prefix, dropping the seed's
        # leading provenance-stage segment. The seed file itself is never edited.
        for k in sorted(kk for kk in e if kk.endswith("_sha256")):
            emitted = "seed_" + k[k.index("_") + 1:]
            L.append(f"{emitted} = {_toml_s(e[k])}")
        L.append("")

    for row in completed_rows + cv_quarantined_rows + no_read_rows + failed_rows:
        L.append("[[entries]]")
        L.append(f"source = {_toml_s(row['source'])}")
        L.append(f"slug = {_toml_s(row['slug'])}")
        L.append(f"language = {_toml_s(row['language'])}")
        L.append(f"predictor_bucket = {_toml_s(row['predictor_bucket'])}")
        L.append(f"completed = {str(row['completed']).lower()}")
        L.append(f"failure_reason = {_toml_s(row['failure_reason'])}")
        L.append(f"ncu_l1_hit_rate_mean = {row['ncu_l1_hit_rate_mean']!r}")
        L.append(f"ncu_l2_hit_rate_mean = {row['ncu_l2_hit_rate_mean']!r}")
        L.append(f"ncu_l1_cv = {row['ncu_l1_cv']!r}")
        L.append(f"ncu_l2_cv = {row['ncu_l2_cv']!r}")
        L.append(f"ncu_run_count = {row['ncu_run_count']}")
        L.append(f"sidecar_path = {_toml_s(row['sidecar_path'])}")
        L.append(f"sidecar_sha256 = {_toml_s(row['sidecar_sha256'])}")
        L.append(f"probe_binary_sha256 = {_toml_s(row['probe_binary_sha256'])}")
        L.append("")

    for row in missing_rows:
        L.append("[[entries]]")
        L.append(f"source = {_toml_s(row['source'])}")
        L.append(f"slug = {_toml_s(row['slug'])}")
        L.append(f"language = {_toml_s(row['language'])}")
        L.append(f"predictor_bucket = {_toml_s(row['predictor_bucket'])}")
        L.append("completed = false")
        L.append(f"failure_reason = {_toml_s(row['failure_reason'])}")
        L.append("")

    out_path.write_text("\n".join(L).rstrip("\n") + "\n", encoding="utf-8")
    print(
        f"wrote {out_path} "
        f"({len(seed_rows)} seed + {len(completed_rows)} completed + "
        f"{len(cv_quarantined_rows)} cv_quarantined + {len(no_read_rows)} no_read + "
        f"{len(failed_rows)} failed + {len(missing_rows)} probe_pending)"
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-only", action="store_true",
                    help="skip the NCU harvest; re-emit the ledger from existing sidecars")
    args = ap.parse_args()
    if not args.emit_only:
        run_harvest(_REPO_ROOT)
    emit_ledger(_REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
