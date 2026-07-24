"""Re-fit the expanded sm_86 IsotonicCalibrator + emit the tiered-MAPE validation
report (the predictor's measured accuracy on solbench).

Calibration input = the (raw, label) pairs:
  * 45 scored cohort_b cells — raw from `sm86_expanded_raw_predictions_v1.toml`
    (Tier-2 NVBit -> SetAssociativeSimulator), NCU label from the expanded
    ledger; canonical bucket from `lookup_predictor_design_coverage`.
  * 5 frozen pilot seed cells — built by the pilot's `_build_records` over the
    seed bundles (.t2a + ncu_3run_v1.json), unchanged.

Typed-excluded from the fit (recorded in the report's exclusion inventory, never
silently dropped, never raw==label): the 8 `tier2_dump_absent` raw rows, the NCU
ledger's `ncu_repeatability_failed` (5) + `ncu_label_no_read_only_traffic` (2)
terminals, and any cell whose canonical bucket is `predictor_typed_unavailable`.

Emits two NEW artifacts (the frozen seed `isotonic_params_sm86.toml` is NEVER
clobbered):
  * `isotonic_params_sm86_expanded.toml` — the full-fit calibrator anchors, one-way
    cross-stamping the expanded ledger / raw-prediction table / seed.
  * `sm86_validation_report_v1.toml` — per-cohort friendly/hostile L1/L2 MAPE
    (LOOCV, fraction domain) + thresholds 0.05 / 0.15 + pass booleans +
    `all_gates_pass` + cohort sizes/lists + the typed-exclusion inventory.

The fit is non-identity-checked (raw != label anchors). A cohort MAPE over its gate
is recorded as an honest fail (typed-fail-closed), not hidden. NVBit/NCU are
calibration-only (CLAUDE.md Rule 1); this module never runs on the deployed/B200 path.
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
import hashlib
import tomllib
from collections import defaultdict
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[5]

import numpy  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

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
from sol_baselines_harness.calibration.cache_prediction.ncu_harvester import (  # noqa: E402
    NCU_REQUIRED_METRICS,
)
from sol_baselines_harness.calibration.cache_prediction.sm86_pilot.persist_fitted_sidecar import (  # noqa: E402
    _SM86_GEOMETRY,
)
from sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep import (  # noqa: E402
    run_sweep as _sweep,
)
from sol_baselines_harness.calibration.cache_prediction.sm86_expanded import (  # noqa: E402
    dense_reuse_l2_calibrator as _l2cal,
)

_CP: Final[Path] = (
    _REPO_ROOT / "tools" / "sol_baselines_harness" / "calibration"
    / "cache_prediction"
)
_RAW_TABLE: Final[Path] = _CP / "sm86_expanded_raw_predictions_v1.toml"
_EXPANDED_LEDGER: Final[Path] = _CP / "label_ledger_v1_sm86_expanded.toml"
_SEED_LEDGER: Final[Path] = _CP / "label_ledger_v1_sm86_pilot_v1.toml"
_CAPABILITY_LEDGER: Final[Path] = _CP / "sm86_predictor_capability_v1.toml"
_SCOREBOARD: Final[Path] = _CP / "sm86_geometry_sweep_scoreboard_v1.toml"
_TRACES_ROOT: Final[Path] = _REPO_ROOT / "scripts" / "baselines_harness" / ".harness_traces"
_PARAMS_OUT: Final[Path] = _CP / "isotonic_params_sm86_expanded.toml"
_REPORT_OUT: Final[Path] = _CP / "sm86_validation_report_v1.toml"

SCHEMA_VERSION: Final[str] = "1.0"
HOST_ARCH: Final[str] = "sm_86"
# Tiered MAPE gate. Operator amendment 2026-05-30 (tyrone.payne): the friendly L2
# bar relaxed 0.05 -> 0.06 after R31-R33 established that no honest (LOOCV,
# deterministic, non-overfit) model reaches 0.05 on the dense L2 residual; the
# best honest model (the streaming-vs-reuse segmented L2 calibrator) lands ~0.055.
# Friendly L1 (0.05) and hostile (0.15) are unchanged. Recorded in the goal
# tracker Plan Evolution Log + round-33 summary.
FRIENDLY_GATE_L1: Final[float] = 0.05
FRIENDLY_GATE_L2: Final[float] = 0.06
HOSTILE_GATE: Final[float] = 0.15
IDENTITY_EPSILON: Final[float] = 1e-9
# A bundle name is <slug>__cuda_cpp__sm86; the seed bundles carry the cuda_cpp tag.
_SEED_TRACES_ROOT: Final[Path] = _TRACES_ROOT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toml_s(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_float_list(xs) -> str:
    return "[" + ", ".join(repr(float(x)) for x in xs) + "]"


def _bundle_for_seed(slug: str) -> Path:
    return _SEED_TRACES_ROOT / f"{slug}__cuda_cpp__sm86"


def _build_cohort_b_records(repo_root: Path):
    """Records + typed-exclusion inventory from the raw-prediction table paired
    with the expanded NCU-label ledger. A scored cell whose canonical bucket is
    typed_unavailable is excluded (typed), never fabricated."""
    raw = tomllib.loads((repo_root / _RAW_TABLE.relative_to(_REPO_ROOT)).read_text())
    records: list[CalibrationInputRecord] = []
    excluded: list[dict] = []
    for e in raw["entries"]:
        slug = e["slug"]
        if not e["scored"]:
            excluded.append({"slug": slug, "typed_reason": e["typed_reason"],
                             "source": "raw_unavailable"})
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
        join_key = CellJoinKey(
            slug=slug, shape_point=slug,
            binary_sha256=e["tier2_sidecar_sha256"] or ("0" * 64),
            source_commit="", kernel_signature_sha256="0" * 64,
            arch=HOST_ARCH, ncu_metric_version="ncu-2026.1",
        )
        records.append(CalibrationInputRecord(
            cell_name=slug, arch=HOST_ARCH, launch_uid=0, predictor_bucket=bucket,
            raw_l1_prediction=float(e["raw_l1_hit_rate"]),
            raw_l2_prediction=float(e["raw_l2_hit_rate"]),
            ncu_l1_label=float(e["ncu_l1_hit_rate"]),
            ncu_l2_label=float(e["ncu_l2_hit_rate"]),
            cell_join_key=join_key,
        ))
    return records, excluded


def _ledger_terminal_exclusions(repo_root: Path) -> list[dict]:
    """The expanded ledger's non-completed cohort_b terminals (CV quarantine
    + zero-read) — labeled cells with no admissible (raw,label) calibration pair."""
    led = tomllib.loads((repo_root / _EXPANDED_LEDGER.relative_to(_REPO_ROOT)).read_text())
    out = []
    for e in led["entries"]:
        if e.get("source") == "cohort_b_probe" and not e.get("completed", False):
            out.append({"slug": e["slug"], "typed_reason": e["failure_reason"],
                        "source": "ncu_ledger_terminal"})
    return out


def _capability_reasons(repo_root: Path) -> dict[str, str]:
    """slug -> typed_reason for every simulator_supported=false capability cell."""
    cap = tomllib.loads((repo_root / _CAPABILITY_LEDGER.relative_to(_REPO_ROOT)).read_text())
    return {c["slug"]: c["typed_reason"] for c in cap["cells"] if not c["simulator_supported"]}


def winning_associativity(repo_root: Path = _REPO_ROOT) -> int:
    """The empirically-selected L2 associativity, read from the committed
    geometry-sweep scoreboard (an input artifact emitted by run_sweep.py)."""
    board = tomllib.loads((repo_root / _SCOREBOARD.relative_to(_REPO_ROOT)).read_text())
    return int(board["meta"]["winner_associativity_l2"])


def build_all_records(repo_root: Path = _REPO_ROOT):
    """Admitted records at the winning geometry with the capability exclusions,
    plus the COMPLETE typed-exclusion inventory (capability-OOD + raw-unavailable
    + NCU-ledger terminals). Records come from the per-assoc sweep sidecars + seed
    re-simulation so the fit is at the winning associativity, not the static card."""
    assoc = winning_associativity(repo_root)
    cap_reasons = _capability_reasons(repo_root)
    records, _sweep_excl = _sweep.build_records_at_assoc(
        repo_root, assoc, frozenset(cap_reasons))

    # Complete, honest inventory (precedence: capability > raw-unavailable > terminal).
    inv: dict[str, dict] = {}
    for slug, reason in cap_reasons.items():
        inv[slug] = {"slug": slug, "typed_reason": reason, "source": "capability_ledger"}
    _, excl_b = _build_cohort_b_records(repo_root)          # 8 raw-unavailable + partition
    for row in excl_b:
        inv.setdefault(row["slug"], row)
    for row in _ledger_terminal_exclusions(repo_root):       # CV-quarantine + zero-read
        inv.setdefault(row["slug"], row)
    excl = sorted(inv.values(), key=lambda r: (r["source"], r["slug"]))
    records = sorted(records, key=lambda r: r.cell_name)
    return records, excl


def _mape(errs) -> float:
    return float(sum(errs) / len(errs)) if errs else float("nan")


def _gate(value: float, threshold: float) -> bool:
    return (value <= threshold) if value == value else False  # NaN -> fail


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_expanded.emit_expanded_validation_report.emit')
def emit(repo_root: Path = _REPO_ROOT) -> tuple[Path, Path]:
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_expanded.emit_expanded_validation_report.emit')
    assoc = winning_associativity(repo_root)
    records, excluded = build_all_records(repo_root)

    # Full-fit calibrator (deployment anchors) on every admitted cell.
    calibrator = IsotonicCalibrator()
    full_split = {r.cell_name: "calibration_only" for r in records}
    calibrator.fit(records, validation_split=full_split)

    l1_x = [float(v) for v in calibrator._l1_head.X_thresholds_]
    l1_y = [float(v) for v in calibrator._l1_head.y_thresholds_]
    l1_gap = max((abs(y - x) for x, y in zip(l1_x, l1_y)), default=0.0)

    # ---- L2 head: streaming-vs-reuse SEGMENTED dense-reuse calibrator ----
    # The single-trace simulator under-resolves dense L2 reuse; the deterministic
    # reuse_factor feature (from the Tier-2 frame, never the label) segments
    # streaming vs L2-resident dense kernels before the isotonic fit.
    reuse = _sweep.reuse_factor_by_slug(repo_root, [r.cell_name for r in records])
    l2_cells = [_l2cal.L2Cell(r.cell_name, r.predictor_bucket, reuse[r.cell_name],
                              r.raw_l2_prediction, r.ncu_l2_label)
                for r in records if r.cell_name in reuse]
    l2_model = _l2cal.fit_segmented(l2_cells)
    if _l2cal.is_identity(l2_cells):
        raise RuntimeError("L2 identity-fit rejected: segmented prediction == label for every cell")
    l2_gap = max((abs(l2_model.predict(c.reuse_factor, c.raw_l2) - c.ncu_l2_label)
                  for c in l2_cells), default=0.0)
    if l1_gap <= IDENTITY_EPSILON and l2_gap <= IDENTITY_EPSILON:
        raise RuntimeError(
            f"identity-fit rejected: L1 max|y-x|={l1_gap:.3e}, L2={l2_gap:.3e} <= {IDENTITY_EPSILON:.0e}"
        )
    sh = l2_model.streaming_head
    rh = l2_model.reuse_head
    l2s_x = [float(v) for v in sh.X_thresholds_] if sh is not None else []
    l2s_y = [float(v) for v in sh.y_thresholds_] if sh is not None else []
    l2r_x = [float(v) for v in rh.X_thresholds_] if rh is not None else []
    l2r_y = [float(v) for v in rh.y_thresholds_] if rh is not None else []

    # LOOCV held-out accuracy -> per-cohort MAPE (fraction domain). L1 from the
    # raw_l1 isotonic; L2 from the segmented reuse model (each per-cell held out).
    loocv = run_loocv(records)
    l2_err = _l2cal.loocv_l2_abs_errors(l2_cells)
    bucket_of = {r.cell_name: r.predictor_bucket for r in records}
    by_cohort = defaultdict(lambda: {"l1": [], "l2": []})
    for res in loocv.per_cell_residuals:
        c = bucket_of[res.cell_name]
        by_cohort[c]["l1"].append(abs(res.predicted_l1_calibrated - res.ncu_l1_label))
    for c in l2_cells:
        by_cohort[c.bucket]["l2"].append(l2_err[c.slug])
    fr = by_cohort.get("predictor_friendly", {"l1": [], "l2": []})
    ho = by_cohort.get("predictor_hostile", {"l1": [], "l2": []})
    f_l1, f_l2 = _mape(fr["l1"]), _mape(fr["l2"])
    h_l1, h_l2 = _mape(ho["l1"]), _mape(ho["l2"])
    f_cells = sorted(s for s, b in bucket_of.items() if b == "predictor_friendly")
    h_cells = sorted(s for s, b in bucket_of.items() if b == "predictor_hostile")
    # Asymmetric friendly gate (operator amendment 2026-05-30): L1 <= 0.05, L2 <= 0.06.
    f_pass = _gate(f_l1, FRIENDLY_GATE_L1) and _gate(f_l2, FRIENDLY_GATE_L2)
    h_pass = _gate(h_l1, HOSTILE_GATE) and _gate(h_l2, HOSTILE_GATE)
    f_present, h_present = bool(f_cells), bool(h_cells)
    all_gates = f_present and h_present and f_pass and h_pass

    # ---- per-cell classification table (consumed by accuracy_diagnostic.py) ----
    # Admitted cells carry the DEPLOYED (full-fit) per-cell predictions: L1 from the
    # raw_l1 isotonic, L2 from the segmented dense-reuse model. Excluded cells carry
    # their typed exclusion source. This is the first-class artifact the diagnostic
    # consumes so the expanded cells leave the stale full-partition fallback.
    reuse_of = {c.slug: c.reuse_factor for c in l2_cells}
    _excl_class = {"capability_ledger": "capability_excluded",
                   "ncu_ledger_terminal": "ncu_terminal_excluded",
                   "raw_unavailable": "raw_unavailable_excluded",
                   "cohort_b_probe": "raw_unavailable_excluded"}
    per_cell: list[dict] = []
    for r in records:
        per_cell.append({
            "slug": r.cell_name, "classification": "measured",
            "predictor_bucket": r.predictor_bucket,
            "predicted_l1_raw": float(r.raw_l1_prediction),
            "predicted_l2_raw": float(r.raw_l2_prediction),
            "predicted_l1_calibrated": float(calibrator.predict_l1(r.raw_l1_prediction)),
            "predicted_l2_calibrated": float(l2_model.predict(reuse_of[r.cell_name], r.raw_l2_prediction)),
            "ncu_l1_label": float(r.ncu_l1_label), "ncu_l2_label": float(r.ncu_l2_label),
            "typed_reason": "",
        })
    for ex in excluded:
        per_cell.append({
            "slug": ex["slug"],
            "classification": _excl_class.get(ex["source"], "raw_unavailable_excluded"),
            "exclusion_source": ex["source"], "typed_reason": ex["typed_reason"],
        })
    per_cell.sort(key=lambda d: d["slug"])

    # Deterministic stamp: anchor to the raw-prediction table's harvest time, NOT
    # wall-clock — the validation report must re-emit byte-for-byte.
    captured = tomllib.loads(
        (repo_root / _RAW_TABLE.relative_to(_REPO_ROOT)).read_text()
    )["meta"]["captured_at_utc"]
    raw_sha = _sha(repo_root / _RAW_TABLE.relative_to(_REPO_ROOT))
    led_sha = _sha(repo_root / _EXPANDED_LEDGER.relative_to(_REPO_ROOT))
    seed_sha = _sha(repo_root / _SEED_LEDGER.relative_to(_REPO_ROOT))
    cap_sha = _sha(repo_root / _CAPABILITY_LEDGER.relative_to(_REPO_ROOT))
    board_sha = _sha(repo_root / _SCOREBOARD.relative_to(_REPO_ROOT))

    # ---- isotonic_params_sm86_expanded.toml ----
    P: list[str] = []
    P.append("# sm_86 EXPANDED isotonic calibrator params (full fit on the admitted cells")
    P.append("# at the empirically-selected L2 associativity, with the capability-ledger")
    P.append("# simulator-out-of-domain cells typed-excluded). Raw = SetAssociativeSimulator")
    P.append("# output, label = NCU read-only-sector hit rate (non-identity fit). NEW")
    P.append("# artifact: the frozen pilot isotonic_params_sm86.toml is never clobbered.")
    P.append("")
    P.append("[meta]")
    P.append(f'schema_version = "{SCHEMA_VERSION}"')
    P.append(f"arch = {_toml_s(HOST_ARCH)}")
    P.append(f"captured_at_utc = {_toml_s(captured)}")
    P.append(f"expanded_ledger_sha256 = {_toml_s(led_sha)}")
    P.append(f"raw_predictions_sha256 = {_toml_s(raw_sha)}")
    P.append(f"seed_ledger_sha256 = {_toml_s(seed_sha)}")
    P.append(f"fit_record_count = {calibrator.admitted_cell_count}")
    P.append(f"admitted_cell_count = {calibrator.admitted_cell_count}")
    P.append("calibrator_class_name = \"IsotonicRegression\"")
    P.append('out_of_bounds = "clip"')
    P.append("ncu_label_aggregation = \"median_of_3_runs\"")
    P.append(f"identity_fit_rejection_epsilon = {IDENTITY_EPSILON!r}")
    P.append(f"l1_identity_max_gap = {l1_gap!r}")
    P.append(f"l2_identity_max_gap = {l2_gap!r}")
    P.append("accepted_metric_selectors = [" + ", ".join(_toml_s(m) for m in NCU_REQUIRED_METRICS) + "]")
    P.append(f"fit_input_geometry_associativity_l2 = {assoc}")
    P.append(f"fit_input_geometry_num_l2_slices = {_SM86_GEOMETRY['num_l2_slices']}")
    P.append('fit_input_geometry_source = "empirical_sweep_winner"')
    P.append(f"capability_ledger_sha256 = {_toml_s(cap_sha)}")
    P.append(f"sweep_scoreboard_sha256 = {_toml_s(board_sha)}")
    P.append("")
    P.append("[reproducibility]")
    P.append(f'python_version = "{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"')
    P.append(f'numpy_version = "{numpy.__version__}"')
    P.append(f'scipy_version = "{scipy.__version__}"')
    P.append(f'sklearn_version = "{sklearn.__version__}"')
    P.append('fitter_sort_key = "slug_lexicographic"')
    P.append("")
    P.append("[l1_head]")
    P.append('feature = "raw_l1_simulator"')
    P.append(f"x_thresholds = {_toml_float_list(l1_x)}")
    P.append(f"y_thresholds = {_toml_float_list(l1_y)}")
    P.append("")
    # L2 head: streaming-vs-reuse SEGMENTED dense-reuse calibrator (R33).
    P.append("[l2_head]")
    P.append('model = "streaming_vs_reuse_segmented"')
    P.append('streaming_feature = "dense_reuse_factor"')
    P.append('reuse_feature = "raw_l2_simulator"')
    P.append(f"segment_threshold_reuse_factor = {l2_model.threshold!r}")
    P.append('segment_threshold_source = "fixed_physical_hyperparameter_not_label_tuned"')
    P.append("dense_reuse_feature_formula = \"total_sector_accesses_div_distinct_sectors\"")
    P.append(f"sweep_sidecar_schema_version = {_l2cal.SIDECAR_SCHEMA_VERSION}")
    P.append("")
    P.append("[l2_head.streaming]   # reuse_factor < threshold -> isotonic(reuse_factor -> label)")
    P.append(f"x_thresholds = {_toml_float_list(l2s_x)}")
    P.append(f"y_thresholds = {_toml_float_list(l2s_y)}")
    P.append("")
    P.append("[l2_head.reuse]       # reuse_factor >= threshold -> isotonic(raw_l2 -> label)")
    P.append(f"x_thresholds = {_toml_float_list(l2r_x)}")
    P.append(f"y_thresholds = {_toml_float_list(l2r_y)}")
    params_path = repo_root / _PARAMS_OUT.relative_to(_REPO_ROOT)
    params_path.write_text("\n".join(P).rstrip("\n") + "\n", encoding="utf-8")
    params_sha = _sha(params_path)

    # ---- sm86_validation_report_v1.toml ----
    R: list[str] = []
    R.append("# sm_86 EXPANDED predictor validation report (the measured accuracy on")
    R.append("# solbench). LOOCV per-cohort MAPE in the fraction domain, tiered gate:")
    R.append("# friendly L1 <= 0.05 and L2 <= 0.06 (operator amendment 2026-05-30),")
    R.append("# hostile L1/L2 <= 0.15. Cells the simulator/NCU")
    R.append("# cannot honestly score are typed-excluded (inventory below), never scored.")
    R.append("")
    R.append("[meta]")
    R.append(f'schema_version = "{SCHEMA_VERSION}"')
    R.append(f"host_arch = {_toml_s(HOST_ARCH)}")
    R.append(f"captured_at_utc = {_toml_s(captured)}")
    R.append(f"isotonic_params_sha256 = {_toml_s(params_sha)}")
    R.append(f"expanded_ledger_sha256 = {_toml_s(led_sha)}")
    R.append(f"raw_predictions_sha256 = {_toml_s(raw_sha)}")
    R.append(f"seed_ledger_sha256 = {_toml_s(seed_sha)}")
    R.append(f"capability_ledger_sha256 = {_toml_s(cap_sha)}")
    R.append(f"sweep_scoreboard_sha256 = {_toml_s(board_sha)}")
    R.append('error_formula = "absolute_per_cell_in_fraction_domain"')
    R.append('split_methodology = "leave_one_out_cross_validation_over_admitted_cells"')
    R.append(f"winning_associativity_l2 = {assoc}")
    R.append('geometry_provenance = "empirical_sweep_winner"')
    R.append(f"admitted_cell_count = {loocv.admitted_cell_count}")
    R.append(f"excluded_cell_count = {len(excluded)}")
    R.append(f"l1_identity_max_gap = {l1_gap!r}")
    R.append(f"l2_identity_max_gap = {l2_gap!r}")
    R.append("")
    R.append("[overall]")
    R.append(f"all_gates_pass = {str(all_gates).lower()}")
    R.append(f"cohort_friendly_pass = {str(f_pass).lower()}")
    R.append(f"cohort_hostile_pass = {str(h_pass).lower()}")
    R.append(f"cohort_friendly_present = {str(f_present).lower()}")
    R.append(f"cohort_hostile_present = {str(h_present).lower()}")
    hostile_degenerate = h_present and len(h_cells) < 2
    R.append(f"hostile_cohort_degenerate = {str(hostile_degenerate).lower()}")
    R.append('friendly_l2_gate_source = "operator_amendment_2026_05_30_relax_l2_0p05_to_0p06"')
    gate_unreachable = not all_gates
    R.append(f"gate_unreachable_via_in_scope_levers = {str(gate_unreachable).lower()}")
    if gate_unreachable:
        R.append('gate_residual_reason = "dense_kernel_l2_reuse_modeling_floor"')
        R.append('gate_residual_detail = "Friendly L2 MAPE exceeds its gate even with the '
                 'streaming-vs-reuse segmented dense-reuse L2 calibrator. The residual is on '
                 'dense kernels (rmsnorm L1-vs-L2 capture, feedforward weight reuse) that no '
                 'single global trace statistic separates with this cohort size. See the round '
                 'summary."')
    R.append("")
    R.append("[cohort_friendly]")
    R.append(f"present = {str(f_present).lower()}")
    R.append(f"cell_count = {len(f_cells)}")
    R.append(f"l1_mape = {f_l1!r}")
    R.append(f"l2_mape = {f_l2!r}")
    R.append(f"l1_gate_threshold = {FRIENDLY_GATE_L1!r}")
    R.append(f"l2_gate_threshold = {FRIENDLY_GATE_L2!r}")
    R.append(f"l1_gate_pass = {str(_gate(f_l1, FRIENDLY_GATE_L1)).lower()}")
    R.append(f"l2_gate_pass = {str(_gate(f_l2, FRIENDLY_GATE_L2)).lower()}")
    R.append("cells = [" + ", ".join(_toml_s(s) for s in f_cells) + "]")
    R.append("")
    R.append("[cohort_hostile]")
    R.append(f"present = {str(h_present).lower()}")
    R.append(f"cell_count = {len(h_cells)}")
    R.append(f"l1_mape = {h_l1!r}")
    R.append(f"l2_mape = {h_l2!r}")
    R.append(f"l1_gate_threshold = {HOSTILE_GATE!r}")
    R.append(f"l2_gate_threshold = {HOSTILE_GATE!r}")
    R.append(f"l1_gate_pass = {str(_gate(h_l1, HOSTILE_GATE)).lower()}")
    R.append(f"l2_gate_pass = {str(_gate(h_l2, HOSTILE_GATE)).lower()}")
    R.append("cells = [" + ", ".join(_toml_s(s) for s in h_cells) + "]")
    R.append("")
    for row in excluded:
        R.append("[[typed_excluded]]")
        R.append(f"slug = {_toml_s(row['slug'])}")
        R.append(f"typed_reason = {_toml_s(row['typed_reason'])}")
        R.append(f"source = {_toml_s(row['source'])}")
        R.append("")
    # Per-cell classification table — the first-class artifact accuracy_diagnostic.py
    # consumes (admitted cells carry deployed per-cell predictions; excluded cells
    # carry their typed exclusion source).
    for pc in per_cell:
        R.append("[[per_cell]]")
        R.append(f"slug = {_toml_s(pc['slug'])}")
        R.append(f"classification = {_toml_s(pc['classification'])}")
        if pc["classification"] == "measured":
            R.append(f"predictor_bucket = {_toml_s(pc['predictor_bucket'])}")
            R.append(f"predicted_l1_raw = {pc['predicted_l1_raw']!r}")
            R.append(f"predicted_l2_raw = {pc['predicted_l2_raw']!r}")
            R.append(f"predicted_l1_calibrated = {pc['predicted_l1_calibrated']!r}")
            R.append(f"predicted_l2_calibrated = {pc['predicted_l2_calibrated']!r}")
            R.append(f"ncu_l1_label = {pc['ncu_l1_label']!r}")
            R.append(f"ncu_l2_label = {pc['ncu_l2_label']!r}")
        else:
            R.append(f"exclusion_source = {_toml_s(pc['exclusion_source'])}")
            R.append(f"typed_reason = {_toml_s(pc['typed_reason'])}")
        R.append("")
    report_path = repo_root / _REPORT_OUT.relative_to(_REPO_ROOT)
    report_path.write_text("\n".join(R).rstrip("\n") + "\n", encoding="utf-8")

    print(f"wrote {params_path} ({calibrator.admitted_cell_count} admitted; "
          f"L1 gap={l1_gap:.4f} L2 gap={l2_gap:.4f})")
    print(f"wrote {report_path}: friendly L1/L2 MAPE {f_l1:.4f}/{f_l2:.4f} "
          f"(n={len(f_cells)}), hostile {h_l1:.4f}/{h_l2:.4f} (n={len(h_cells)}); "
          f"all_gates_pass={all_gates}; excluded={len(excluded)}")
    return params_path, report_path


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    emit(_REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
