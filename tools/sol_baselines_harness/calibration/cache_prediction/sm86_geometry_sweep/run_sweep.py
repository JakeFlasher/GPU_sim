"""sm_86 L2-associativity sweep + winner selection (keystone predictor accuracy-gate path).

For each candidate `associativity_l2 in {4,8,16,32,64}`, build the full
calibration record set (45 cohort_b cells from the sweep sidecars + 5 seed cells
re-simulated from their on-disk Tier-2 traces — both at the SAME associativity for
geometric consistency), fit the IsotonicCalibrator, run leave-one-out CV, and
compute fraction-domain per-cohort/head MAPE. Select the associativity whose fit
best satisfies the tiered gate (friendly L1∧L2 ≤ 0.05, hostile L1∧L2 ≤ 0.15).

`build_records_at_assoc(repo, assoc, exclude)` is the reusable record builder the
final emitter also calls so the winning geometry's fit is reproduced byte-for-byte.
The empirical-sweep result is MODEL SELECTION, not authoritative hardware geometry
(BL-20260526-empirical-geometry-sweep-with-trace-replay).
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
import dataclasses
import json
import statistics
import tomllib
from collections import defaultdict
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[5]

from sol_baselines_harness.calibration.cache_prediction import (
    historical_path_aliases,
)
from sol_baselines_harness.metrics.cache_predictor import predict_from_launch_frame  # noqa: E402
from sol_baselines_harness.metrics.tier2_sidecar_parser import parse_launch_frames  # noqa: E402
from sol_baselines_harness.metrics.isotonic_calibrator import (  # noqa: E402
    CalibrationInputRecord,
    CellJoinKey,
    IsotonicCalibrator,
    run_loocv,
)
from sol_baselines_harness.metrics.predictor_design_coverage import (  # noqa: E402
    lookup_predictor_design_coverage,
    PredictorDesignCoverageError,
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
)

_CP: Final[Path] = (
    _REPO_ROOT / "tools" / "sol_baselines_harness" / "calibration"
    / "cache_prediction"
)
_SWEEP_SIDECARS: Final[Path] = _CP / "raw_pred_sweep_sidecars_sm86"
_EXPANDED_LEDGER: Final[Path] = _CP / "label_ledger_v1_sm86_expanded.toml"
_SEED_LEDGER: Final[Path] = _CP / "label_ledger_v1_sm86_pilot_v1.toml"


def _resolve_seed_artifact(
    repo_root: Path,
    entry: dict,
    filename: str,
) -> Path:
    expected_hash_field = {
        "ncu_3run_v1.json": "task9_ncu_3run_v1_sha256",
        "nvbit_metrics/tier2_address_trace.t2a": (
            "task10_tier2_binary_sha256"
        ),
    }.get(filename)
    if expected_hash_field is None:
        raise ValueError(f"unsupported retained seed artifact: {filename}")
    historical_path = (
        "scripts/baselines_harness/.harness_traces/"
        f"{entry['slug']}__cuda_cpp__sm86/{filename}"
    )
    return historical_path_aliases.resolve_historical_file(
        historical_path,
        alias_id=historical_path_aliases.SM86_HARNESS_TRACES_ALIAS,
        expected_sha256=entry[expected_hash_field],
        repo_root=repo_root,
    )


HOST_ARCH: Final[str] = "sm_86"
FRIENDLY_GATE: Final[float] = 0.05
HOSTILE_GATE: Final[float] = 0.15
CANDIDATES: Final[tuple[int, ...]] = (4, 8, 16, 32, 64)


def _geometry_spec(assoc: int) -> dict:
    return {
        "arch": "sm_86", "l1_kb_per_sm": 192, "l1_kb_per_sm_source": "static_device_card",
        "l2_capacity_bytes": 6 * 1024 * 1024, "capacity_source": "static_device_card",
        "num_l2_slices": 6, "num_l2_slices_source": "static_device_card",
        "sector_size_bytes": 32, "sector_size_source": "static_device_card",
        "line_size_bytes": 128, "line_size_source": "static_device_card",
        "associativity_l2": assoc, "associativity_l2_source": "empirical_sweep_candidate",
        "l2_constant_memory_counts_for_read_score": False,
        "l2_constant_memory_counts_for_read_score_source": "static_device_card_sm_86",
        "geometry_source": "empirical_sweep_candidate",
    }


def _median_ncu(ncu_path: Path) -> tuple[float, float]:
    data = json.loads(ncu_path.read_text())
    runs = data.get("runs", [])
    l1s = [float(r["l1_global_load_hit_rate"]) for r in runs]
    l2s = [float(r["l2_read_hit_rate"]) for r in runs]
    return statistics.median(l1s), statistics.median(l2s)


def _record(slug: str, bucket: str, raw_l1: float, raw_l2: float,
            ncu_l1: float, ncu_l2: float, sha: str) -> CalibrationInputRecord:
    return CalibrationInputRecord(
        cell_name=slug, arch=HOST_ARCH, launch_uid=0, predictor_bucket=bucket,
        raw_l1_prediction=float(raw_l1), raw_l2_prediction=float(raw_l2),
        ncu_l1_label=float(ncu_l1), ncu_l2_label=float(ncu_l2),
        cell_join_key=CellJoinKey(
            slug=slug, shape_point=slug, binary_sha256=sha or ("0" * 64),
            source_commit="", kernel_signature_sha256="0" * 64,
            arch=HOST_ARCH, ncu_metric_version="ncu-2026.1"),
    )


def reuse_factor_by_slug(repo_root: Path, slugs) -> dict:
    """Deterministic dense-reuse factor per slug: cohort_b from the sweep sidecars
    (schema v2 `dense_reuse`), seed cells re-derived from their on-disk `.t2a`."""
    from sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep import (
        harvest_sweep_traces as _hsv,
    )
    sidecar_dir = repo_root / _SWEEP_SIDECARS.relative_to(_REPO_ROOT)
    seed_document = tomllib.loads(
        (repo_root / _SEED_LEDGER.relative_to(_REPO_ROOT)).read_text()
    )
    seed_entries = _index_unique_entries(
        seed_document["entries"],
        ledger_name="pilot sm86 seed ledger",
    )
    out: dict = {}
    want = set(slugs)
    for slug in want:
        sc = sidecar_dir / f"{slug}.json"
        if sc.is_file():
            d = json.loads(sc.read_text())
            dr = d.get("dense_reuse")
            if dr:
                out[slug] = float(dr["reuse_factor"])
                continue
        seed_entry = seed_entries.get(slug)
        if seed_entry is None:
            continue
        t2a = _resolve_seed_artifact(
            repo_root,
            seed_entry,
            "nvbit_metrics/tier2_address_trace.t2a",
        )
        out[slug] = float(
            _hsv.dense_reuse_feature(parse_launch_frames(t2a)[0])[
                "reuse_factor"
            ]
        )
    return out


_COHORT_FAILURE_REASONS: Final[frozenset[str]] = frozenset({
    "ncu_label_no_read_only_traffic",
    "ncu_repeatability_failed",
})
_SEED_CROSS_STAMPS: Final[dict[str, str]] = {
    "task10_tier2_binary_sha256": "seed_tier2_binary_sha256",
    "task10_tier2_v1_sha256": "seed_tier2_v1_sha256",
    "task8_full_correctness_v1_sha256": "seed_full_correctness_v1_sha256",
    "task9_ncu_3run_v1_sha256": "seed_ncu_3run_v1_sha256",
}


def _index_unique_entries(entries, *, ledger_name: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{ledger_name}: ledger entry must be a table")
        slug = entry.get("slug")
        if not isinstance(slug, str) or not slug:
            raise ValueError(f"{ledger_name}: ledger entry has invalid slug")
        if slug in indexed:
            raise ValueError(f"{ledger_name}: duplicate slug: {slug}")
        indexed[slug] = entry
    return indexed


def _cohort_failure_reason(slug: str, entry: dict) -> str | None:
    completed = entry.get("completed")
    failure_reason = entry.get("failure_reason")
    if type(completed) is not bool:
        raise ValueError(
            f"expanded sm86 ledger: {slug}: completed must be boolean"
        )
    if completed:
        if failure_reason != "":
            raise ValueError(
                f"expanded sm86 ledger: {slug}: completed row has "
                "failure_reason"
            )
        return None
    if failure_reason not in _COHORT_FAILURE_REASONS:
        raise ValueError(
            f"expanded sm86 ledger: {slug}: unsupported "
            f"failure_reason: {failure_reason!r}"
        )
    return failure_reason


def _validate_seed_cross_stamps(
    slug: str,
    pilot_entry: dict,
    expanded_entry: dict,
) -> None:
    if type(expanded_entry.get("completed")) is not bool:
        raise ValueError(
            f"expanded sm86 ledger: {slug}: completed must be boolean"
        )
    if not expanded_entry["completed"]:
        raise ValueError(
            f"expanded sm86 ledger: {slug}: seed row is incomplete"
        )
    if expanded_entry.get("failure_reason") != "":
        raise ValueError(
            f"expanded sm86 ledger: {slug}: completed row has "
            "failure_reason"
        )
    for pilot_field, expanded_field in _SEED_CROSS_STAMPS.items():
        pilot_sha256 = pilot_entry.get(pilot_field)
        expanded_sha256 = expanded_entry.get(expanded_field)
        for field, value in (
            (pilot_field, pilot_sha256),
            (expanded_field, expanded_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise ValueError(
                    f"sm86 ledgers: {slug}: invalid SHA-256 field: {field}"
                )
        if pilot_sha256 != expanded_sha256:
            raise ValueError(
                f"sm86 ledgers: {slug}: seed cross-stamp mismatch: "
                f"{pilot_field} != {expanded_field}"
            )


def build_records_at_assoc(repo_root: Path, assoc: int, exclude=frozenset()):
    """All admissible (raw@assoc, ncu) records for one associativity, plus the
    typed-exclusion inventory. `exclude` adds capability-grounded slugs."""
    geom = _geometry_spec(assoc)
    sidecar_dir = repo_root / _SWEEP_SIDECARS.relative_to(_REPO_ROOT)
    led = tomllib.loads((repo_root / _EXPANDED_LEDGER.relative_to(_REPO_ROOT)).read_text())
    expanded_entries = _index_unique_entries(
        led["entries"],
        ledger_name="expanded sm86 ledger",
    )
    unexpected_sources = {
        entry.get("source") for entry in expanded_entries.values()
    } - {"cohort_b_probe", "seed"}
    if unexpected_sources:
        raise ValueError(
            "expanded sm86 ledger: unsupported sources: "
            f"{sorted(unexpected_sources, key=repr)!r}"
        )
    cohort_entries = {
        slug: entry for slug, entry in expanded_entries.items()
        if entry["source"] == "cohort_b_probe"
    }
    labels: dict[str, dict] = {}
    records: list[CalibrationInputRecord] = []
    excluded: list[dict] = []
    for slug, entry in sorted(cohort_entries.items()):
        failure_reason = _cohort_failure_reason(slug, entry)
        if failure_reason is None:
            labels[slug] = entry
            continue
        excluded.append({
            "slug": slug,
            "typed_reason": failure_reason,
            "source": "cohort_b_probe",
        })

    # --- cohort_b cells: per-assoc raw from the sweep sidecars ---
    for slug in sorted(labels):
        sc_path = sidecar_dir / f"{slug}.json"
        if not sc_path.is_file():
            # Fail closed: a labeled cohort_b cell whose per-assoc sweep
            # sidecar is missing on disk is surfaced as a typed exclusion,
            # NOT silently dropped from both the record set and the
            # inventory. A silent ``continue`` here would let an
            # incomplete harvest masquerade as a smaller-but-complete
            # calibration cohort and bias the gate.
            excluded.append({"slug": slug,
                             "typed_reason": "tier2_sidecar_missing",
                             "source": "cohort_b_probe"})
            continue
        sc = json.loads(sc_path.read_text())
        a = sc.get("by_assoc", {}).get(str(assoc), {})
        if not a.get("scored"):
            excluded.append({"slug": slug, "typed_reason": "tier2_dump_absent",
                             "source": "cohort_b_probe"})
            continue
        try:
            bucket = lookup_predictor_design_coverage(slug).predictor_bucket
        except (PredictorDesignCoverageError, KeyError):
            excluded.append({"slug": slug, "typed_reason": "cell_missing_cohort_partition",
                             "source": "cohort_b_probe"})
            continue
        if bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
            excluded.append({"slug": slug, "typed_reason": "predictor_typed_unavailable",
                             "source": "cohort_b_probe"})
            continue
        if slug in exclude:
            excluded.append({"slug": slug, "typed_reason": "predictor_typed_unavailable",
                             "source": "capability_ledger"})
            continue
        records.append(_record(slug, bucket, a["raw_l1_hit_rate"], a["raw_l2_hit_rate"],
                               labels[slug]["ncu_l1_hit_rate_mean"],
                               labels[slug]["ncu_l2_hit_rate_mean"],
                               sc.get("tier2_sidecar_sha256", "")))

    # --- seed cells: re-simulate the on-disk trace at this assoc ---
    seed = tomllib.loads((repo_root / _SEED_LEDGER.relative_to(_REPO_ROOT)).read_text())
    seed_entries = _index_unique_entries(
        seed["entries"],
        ledger_name="pilot sm86 seed ledger",
    )
    expanded_seed_slugs = {
        slug for slug, entry in expanded_entries.items()
        if entry["source"] == "seed"
    }
    if set(seed_entries) != expanded_seed_slugs:
        raise ValueError(
            "sm86 ledgers: expanded and pilot seed slug sets differ"
        )
    if set(seed_entries) & set(cohort_entries):
        raise ValueError("sm86 ledgers: seed and cohort slug sets overlap")
    for slug, pilot_entry in seed_entries.items():
        _validate_seed_cross_stamps(
            slug,
            pilot_entry,
            expanded_entries[slug],
        )
    for e in seed_entries.values():
        slug = e["slug"]
        t2a = _resolve_seed_artifact(
            repo_root,
            e,
            "nvbit_metrics/tier2_address_trace.t2a",
        )
        ncu = _resolve_seed_artifact(
            repo_root,
            e,
            "ncu_3run_v1.json",
        )
        try:
            bucket = lookup_predictor_design_coverage(slug).predictor_bucket
        except (PredictorDesignCoverageError, KeyError):
            excluded.append({
                "slug": slug,
                "typed_reason": "cell_missing_cohort_partition",
                "source": "seed",
            })
            continue
        if bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
            excluded.append({
                "slug": slug,
                "typed_reason": "predictor_typed_unavailable",
                "source": "seed",
            })
            continue
        frames = parse_launch_frames(t2a)
        if not frames:
            excluded.append({"slug": slug, "typed_reason": "tier2_parse_no_frames",
                             "source": "seed"})
            continue
        frame = frames[0]
        if frame.truncation is not None:
            frame = dataclasses.replace(frame, truncation=None)
        result = predict_from_launch_frame(frame, cell_name=slug, geometry_spec=geom)
        if result.get("result_kind") != "scalar":
            excluded.append({"slug": slug, "typed_reason": "predictor_typed_unavailable",
                             "source": "seed"})
            continue
        if slug in exclude:
            excluded.append({"slug": slug, "typed_reason": "predictor_typed_unavailable",
                             "source": "capability_ledger"})
            continue
        ncu_l1, ncu_l2 = _median_ncu(ncu)
        import hashlib as _hl
        seed_sha = _hl.sha256(t2a.read_bytes()).hexdigest()
        records.append(_record(slug, bucket, result["raw_l1_hit_rate"],
                               result["raw_l2_hit_rate"], ncu_l1, ncu_l2, seed_sha))

    records.sort(key=lambda r: r.cell_name)
    excluded.sort(key=lambda r: (r["source"], r["slug"]))
    return records, excluded


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep.run_sweep.evaluate_assoc')
def evaluate_assoc(repo_root: Path, assoc: int, exclude=frozenset()) -> dict:
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep.run_sweep.evaluate_assoc')
    records, excluded = build_records_at_assoc(repo_root, assoc, exclude)
    bucket_of = {r.cell_name: r.predictor_bucket for r in records}
    calib = IsotonicCalibrator()
    calib.fit(records, validation_split={r.cell_name: "calibration_only" for r in records})
    loocv = run_loocv(records)
    by = defaultdict(lambda: {"l1": [], "l2": []})
    for res in loocv.per_cell_residuals:
        c = bucket_of[res.cell_name]
        by[c]["l1"].append(abs(res.predicted_l1_calibrated - res.ncu_l1_label))
        by[c]["l2"].append(abs(res.predicted_l2_calibrated - res.ncu_l2_label))

    def mape(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")
    fr, ho = by.get("predictor_friendly", {"l1": [], "l2": []}), by.get("predictor_hostile", {"l1": [], "l2": []})
    f_l1, f_l2, h_l1, h_l2 = mape(fr["l1"]), mape(fr["l2"]), mape(ho["l1"]), mape(ho["l2"])
    f_present, h_present = bool(fr["l1"]), bool(ho["l1"])

    def g(v, t):
        return (v <= t) if v == v else False
    f_pass = f_present and g(f_l1, FRIENDLY_GATE) and g(f_l2, FRIENDLY_GATE)
    h_pass = h_present and g(h_l1, HOSTILE_GATE) and g(h_l2, HOSTILE_GATE)
    return {
        "associativity_l2": assoc, "admitted": len(records), "excluded": len(excluded),
        "friendly_n": len(fr["l1"]), "hostile_n": len(ho["l1"]),
        "friendly_l1_mape": f_l1, "friendly_l2_mape": f_l2,
        "hostile_l1_mape": h_l1, "hostile_l2_mape": h_l2,
        "friendly_gate_pass": f_pass, "hostile_gate_pass": h_pass,
        "all_gates_pass": bool(f_present and h_present and f_pass and h_pass),
    }


_CAPABILITY_LEDGER: Final[Path] = _CP / "sm86_predictor_capability_v1.toml"
_SCOREBOARD_OUT: Final[Path] = _CP / "sm86_geometry_sweep_scoreboard_v1.toml"


def load_capability_unsupported(repo_root: Path = _REPO_ROOT) -> frozenset:
    """The capability-ledger slugs whose dominant access is indirect /
    data-dependent (simulator_supported = false) — typed-excluded from the gate
    cohort. Grounded in kernel algorithm, NOT residual."""
    cap = tomllib.loads((repo_root / _CAPABILITY_LEDGER.relative_to(_REPO_ROOT)).read_text())
    return frozenset(c["slug"] for c in cap["cells"] if not c["simulator_supported"])


def select_winner(results_excluded: list[dict]) -> dict:
    """Pick the associativity that best satisfies the tiered gate: prefer a
    gates-passing candidate; else the candidate with the lowest aggregate
    fraction-domain MAPE (deterministic; ties broken by smaller associativity)."""
    refuse_legacy_ncu_calibration('sm86_geometry_sweep.run_sweep.select_winner')
    passing = [r for r in results_excluded if r["all_gates_pass"]]
    pool = passing or results_excluded

    def agg(r):
        vals = [r["friendly_l1_mape"], r["friendly_l2_mape"],
                r["hostile_l1_mape"], r["hostile_l2_mape"]]
        return sum(v for v in vals if v == v)  # NaN-safe
    return min(pool, key=lambda r: (agg(r), r["associativity_l2"]))


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep.run_sweep.emit_scoreboard')
def emit_scoreboard(repo_root: Path = _REPO_ROOT) -> Path:
    """Deterministic sweep scoreboard: per-assoc MAPE both BARE (geometry-only)
    and WITH the capability ledger's typed exclusions, plus the winner. The
    evidence that geometry alone floors friendly L2, and that even with the honest
    capability exclusions the dense-kernel residual keeps friendly L2 above 0.05."""
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep.run_sweep.emit_scoreboard')
    unsup = load_capability_unsupported(repo_root)
    bare = [evaluate_assoc(repo_root, a, frozenset()) for a in CANDIDATES]
    excl = [evaluate_assoc(repo_root, a, unsup) for a in CANDIDATES]
    winner = select_winner(excl)
    cap_sha = _sha(repo_root / _CAPABILITY_LEDGER.relative_to(_REPO_ROOT))

    def _q(v: str) -> str:
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def _block(tag: str, rows: list[dict]) -> list[str]:
        out = []
        for r in rows:
            out.append(f"[[{tag}]]")
            out.append(f"associativity_l2 = {r['associativity_l2']}")
            out.append(f"admitted = {r['admitted']}")
            out.append(f"friendly_n = {r['friendly_n']}")
            out.append(f"hostile_n = {r['hostile_n']}")
            out.append(f"friendly_l1_mape = {r['friendly_l1_mape']!r}")
            out.append(f"friendly_l2_mape = {r['friendly_l2_mape']!r}")
            out.append(f"hostile_l1_mape = {r['hostile_l1_mape']!r}")
            out.append(f"hostile_l2_mape = {r['hostile_l2_mape']!r}")
            out.append(f"all_gates_pass = {str(r['all_gates_pass']).lower()}")
            out.append("")
        return out

    L: list[str] = []
    L.append("# sm_86 L2-associativity geometry-sweep scoreboard (predictor accuracy-gate evidence).")
    L.append("# GENERATED (deterministic) by sm86_geometry_sweep/run_sweep.py.")
    L.append("# `bare_*` = geometry sweep only (no exclusions); `excluded_*` = with the")
    L.append("# capability-ledger typed exclusions. Empirical model SELECTION, not an")
    L.append("# authoritative hardware fact (BL-20260526-empirical-geometry-sweep).")
    L.append("")
    L.append("[meta]")
    L.append("schema_version = 1")
    L.append('host_arch = "sm_86"')
    L.append(f"candidates = {list(CANDIDATES)!r}".replace("(", "[").replace(")", "]"))
    L.append(f"capability_ledger_sha256 = {_q(cap_sha)}")
    L.append(f"winner_associativity_l2 = {winner['associativity_l2']}")
    L.append(f"winner_all_gates_pass = {str(winner['all_gates_pass']).lower()}")
    L.append('winner_selection = "min_aggregate_fraction_mape_then_smaller_assoc"')
    L.append('geometry_provenance = "empirical_sweep_winner"')
    L.append("")
    L += _block("bare_candidate", bare)
    L += _block("excluded_candidate", excl)
    out_path = repo_root / _SCOREBOARD_OUT.relative_to(_REPO_ROOT)
    out_path.write_text("\n".join(L).rstrip("\n") + "\n", encoding="utf-8")
    print(f"wrote {out_path} (winner assoc={winner['associativity_l2']}, "
          f"all_gates_pass={winner['all_gates_pass']})")
    return out_path


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep.run_sweep.sweep')
def sweep(repo_root: Path = _REPO_ROOT, exclude=frozenset()) -> list[dict]:
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep.run_sweep.sweep')
    out = []
    for assoc in CANDIDATES:
        r = evaluate_assoc(repo_root, assoc, exclude)
        out.append(r)
        print(f"  assoc={assoc:2d}  friendly L1/L2 {r['friendly_l1_mape']:.4f}/{r['friendly_l2_mape']:.4f}"
              f"  hostile L1/L2 {r['hostile_l1_mape']:.4f}/{r['hostile_l2_mape']:.4f}"
              f"  gates_pass={r['all_gates_pass']}  (admitted={r['admitted']})", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    sweep(_REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
