#!/usr/bin/env python3
"""Generate every figure for the final project report from the measured results.

Run:  ../.paper_venv/bin/python figures/gen_figures.py

Design goals (rewritten 2026-06-09 for an expert readership of conference chairs):
every figure must (a) tell ONE plain-language story, (b) carry real measured
information rather than decorative bars, and (c) never visually contradict a
number asserted elsewhere in the report. Measured constants are transcribed from
the project artifacts (source noted per figure) and re-asserted against them in
_verify_against_artifacts() -- any drift raises AssertionError. When the raw
artifacts are absent (e.g. a stripped tree) the checks skip gracefully so the
figures still build.

Scope notes carried over from the project:
* The identifiability map presents the 11 essential summary-page metrics
  (3 recoverable + 8 non-identifiable, all certified) in the MAIN BODY; the
  registered adjudication covers 20 (4 + 16, one certificate deferred), the 8
  near-duplicate variants are relegated to the artifacts, and wall-clock
  kernel time is reported only as a limitation. The trivially-exact counting
  metrics are out of scope.
* RTX 5060 (Blackwell, sm_120) is the primary measured GPU; the RTX 3090 Ti
  (Ampere, sm_86) cross-architecture study is complete. The recoverable traffic
  estimates and the timing story are cross-arch confirmed. The large-cache (L2)
  cross-arch verdict is STRUCTURAL: the erasure mechanism is identical on both
  architectures; a calibrated sm_86 L2 head reaches the aggregate gate but is
  retained only as a fitted aggregate (it fails leave-one-out robustness), so no
  fitted per-family Ampere value is asserted.
* The per-genre / per-card availability counts in fig_framework are final
  benchmark availability counts.
"""
import os, sys, json, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
import plotstyle as ps

ps.apply_style()
plt.rcParams["pdf.fonttype"] = 42  # TrueType: silences font-type warnings in PDF checkers
OUT = os.path.dirname(os.path.abspath(__file__))
C = ps.PALETTE
TIER = ps.TIER_COLOR

# Raw committed artifacts (present on the dev machine). Headline constants are
# asserted against them so a figure can never silently drift from the data.
_ART = os.path.normpath(os.path.join(OUT, "..", "..", "5060_branch",
                                     "data", "local_labels"))
_AMP = os.path.normpath(os.path.join(OUT, "..", "..", "3090ti_branch", "data"))
_SB86 = os.path.normpath(os.path.join(OUT, "..", "..", "3090ti_solbench",
                                      "scripts", "baselines_harness",
                                      "calibration", "cache_prediction"))
_SB120 = os.path.normpath(os.path.join(OUT, "..", "..", "5060_solbench",
                                       "scripts", "baselines_harness",
                                       "calibration", "cache_prediction"))


def _load(p, base=_ART):
    fp = os.path.join(base, p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def _load_toml(p, base=None):
    fp = os.path.join(base, p) if base else p
    return tomllib.load(open(fp, "rb")) if os.path.exists(fp) else None


# Genre identity for the SOL-ExecBench suite figures (benchmark difficulty
# tiers L1/L2 + Quant + FIB). Deliberately disjoint from the two reserved
# verdict hues (#0072B2 / #D55E00) and from the GPU teal/slate pair.
GENRE_COLOR = {"L1": C["sky"], "L2": C["purple"],
               "Quant": C["yellow"], "FIB": C["green"]}


# Idle P-state floor logged while the GPU was deadlocked (user-flagged
# 2026-06-10): the three 180.0 MHz cells (flash_attn_fwd x2, gemm_hmma_bf16 x1,
# mirrored in both timing artifacts) were sampled while no compute was in
# flight, so they are not DVFS states reached under load and are excluded
# from the plotted clock swarm. The raw artifacts keep them (read-only).
DEADLOCK_IDLE_MHZ = 180.0


def _clock_samples():
    """NVML SM-clock samples behind fig_timing, deadlock-idle readings
    excluded. Returns (sorted array or None, number of excluded samples)."""
    clocks = []
    da = _load("duration_admitted.json")
    if da:
        for c in da.get("cells", []):
            v = (c.get("nvml_covariate_summary") or {}).get("sm_clock_mhz_median")
            if v is not None:
                clocks.append(v)
    tw = _load("sm120_timing_cupti_ncu_crosswalk.json")
    if tw:
        clocks += [c["nvml_sm_clock_mhz"] for c in tw["cells"]
                   if c.get("nvml_sm_clock_mhz") is not None]
    if not clocks:
        return None, 0
    keep = [v for v in clocks if abs(v - DEADLOCK_IDLE_MHZ) >= 0.5]
    return np.array(sorted(keep)), len(clocks) - len(keep)


# ---------------------------------------------------------------------------
# SOL-ExecBench suite accuracy (fig_framework panels b-d) -- the deployed
# cache-prediction surface scored against the locally produced NCU answer key
# on the benchmark's own problems. Single source of truth for BOTH the figure
# and the guard. Measured constants come from the harness's own registered
# validation reports (LOOCV cohort gates); the per-cell points re-apply the
# FROZEN deployed calibrator anchors (isotonic_params_*.toml) to the persisted
# raw replay outputs via pure interpolation -- no pipeline re-fit.
# ---------------------------------------------------------------------------
def _genre_map():
    """slug -> genre (L1/L2/Quant/FIB) from the baselines template trees of
    both solbench repos (no slug appears under two genres; verified)."""
    g = {}
    for repo in ("5060_solbench", "3090ti_solbench"):
        root = os.path.normpath(os.path.join(OUT, "..", "..", repo, "baselines"))
        for genre in ("L1", "L2", "Quant", "FIB"):
            d = os.path.join(root, genre)
            if os.path.isdir(d):
                for slug in os.listdir(d):
                    if os.path.isdir(os.path.join(d, slug)):
                        g.setdefault(slug, genre)
    return g


# The single fitted value of the sm_86 L2 head's reuse branch (its isotonic
# anchors are a constant): every staged-reuse cell receives this prediction.
# Disclosed in the figure and caption as a family prior, not a recovery.
_L2_REUSE_PLATEAU = 0.9114212787532042


def _suite_data():
    """Returns (cells, rep86, rep120, ladder_counts) or (None,)*4.
    cells: per-problem dicts {slug, genre, bucket, ncu_l1, cal_l1, ncu_l2,
    cal_l2} for the FULL sm_86 admitted cohort (35 = 34 friendly + 1
    hostile), read straight from the validation report's own per-cell
    record of the deployed calibrated predictions."""
    rep86 = _load_toml("sm86_validation_report_v1.toml", _SB86)
    rep120 = _load_toml("sm120_validation_report_v1.toml", _SB120)
    lad = _load_toml("sm120_coverage_readiness_ladder_v1.toml", _SB120)
    ladder = None
    if lad:
        ladder = {}
        for e in lad["entries"]:
            ladder[e["state"]] = ladder.get(e["state"], 0) + 1
    if not rep86:
        return None, rep86, rep120, ladder
    admitted = set(rep86["cohort_friendly"]["cells"]) | set(rep86["cohort_hostile"]["cells"])
    pc = {e["slug"]: e for e in rep86["per_cell"]}
    genre = _genre_map()
    cells = [dict(slug=s, genre=genre.get(s, "?"),
                  bucket=pc[s]["predictor_bucket"],
                  ncu_l1=pc[s]["ncu_l1_label"], cal_l1=pc[s]["predicted_l1_calibrated"],
                  ncu_l2=pc[s]["ncu_l2_label"], cal_l2=pc[s]["predicted_l2_calibrated"])
             for s in sorted(admitted)]
    return cells, rep86, rep120, ladder


# ---------------------------------------------------------------------------
# Microbenchmark per-cell (prediction, NCU label) pairs (fig_microbench) --
# loaded straight from the released per-cell artifacts; nothing recomputed
# beyond elementwise relative error.
# ---------------------------------------------------------------------------
def _sm120_traffic_rows():
    """Full sm_120 trace-cohort raw traffic recovery: list of
    (read_label, raw_read, write_label, unique_store) rows (n=158)."""
    d = _load("sm120_xbar_head_cohort_diagnostic.json")
    if not d:
        return None
    rows = []
    for r in d["rows"]:
        if r.get("read_unavailable_reason") is None and r.get("xbar_read_label") is not None:
            rows.append((float(r["xbar_read_label"]), float(r["raw_read_sectors"])))
    return rows


def _t2_pairs(tp, name):
    """Scored (label, predicted) pairs for one primitive of the sm_86
    traffic-prediction artifact, with the slug kept for outlier labelling."""
    return [(c["slug"], float(p["label"]), float(p["predicted"]))
            for c in tp["cells"] for p in c.get("primitives", [])
            if p.get("primitive") == name and p.get("status") == "scored"]


def _sm86_l1_cells():
    d = _load(os.path.join("local_labels", "l1_closure_comparison_sm_86.json"),
              base=_AMP)
    if not d:
        return None
    return [(c["ground_truth_fraction"], c["predicted_hit_rate"])
            for c in d["included_cells"]]


def _sm120_l2_cells():
    d = _load("sm120_l2_tree_plru_model.json")
    if not d:
        return None
    return [(c["family_bucket"], c["label"], c["predicted"])
            for c in d["cells"]
            if c.get("evaluated") and c.get("abs_error") is not None]


# ---------------------------------------------------------------------------
# The essential-metric identifiability census: the 11 summary-page metrics a
# performance analyst actually reads, drawn from the registered 20-metric
# adjudication. Each row:
#   group | short human label | exact NCU metric id | verdict | status
#   verdict: "rec" (recoverable) | "non" (non-identifiable)
#   status:  "accepted"  -- recoverable, recovery accepted (a positive estimate)
#            "outside_support" -- recoverable tier, estimate not accepted by its gate
#            "erased"    -- non-identifiable, falsifiable certificate issued
# Counts (3 recoverable = 2 accepted + 1 outside support; 8 non-identifiable, all
# certified) and every per-metric tier are asserted against
# recoverable_noniden_closure.json, as is the partition: essential 11 +
# VARIANTS_NOT_PLOTTED 8 + NOT_PLOTTED_TIME 1 = the registered 20 (4/16).
# ---------------------------------------------------------------------------
CENSUS = [
    ("Cache traffic\n(L1↔L2)", "L1→L2 read",
     "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum",  "rec", "accepted", False),
    ("Cache traffic\n(L1↔L2)", "L1→L2 write",
     "l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum", "rec", "accepted", False),
    ("Cache hit\nrate",        "L1 hit rate",
     "l1tex__t_sector_hit_rate.pct",                       "rec", "outside_support", False),
    ("Cache hit\nrate",        "L2 hit ratio",
     "lts__t_sector_hit_rate.pct",                         "non", "erased",   True),
    ("DRAM\ntraffic",          "DRAM read",
     "dram__sectors_op_read.sum",                          "non", "erased",   False),
    ("DRAM\ntraffic",          "DRAM write",
     "dram__sectors_op_write.sum",                         "non", "erased",   False),
    ("DRAM\ntraffic",          "DRAM bandwidth",
     "dram__bytes.sum.per_second",                         "non", "erased",   False),
    ("Cycles",                 "elapsed cycles",
     "gpc__cycles_elapsed.max",                            "non", "erased",   False),
    ("Occupancy",              "achieved\noccupancy",
     "sm__warps_active.avg.pct_of_peak_sustained_active",  "non", "erased",   False),
    ("Throughput\n& IPC",      "memory\nthroughput",
     "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed", "non", "erased", False),
    ("Throughput\n& IPC",      "achieved IPC",
     "sm__inst_executed.avg.per_cycle_elapsed",            "non", "erased",   False),
]

# The registered list's 8 near-duplicate variants are trimmed from the paper
# because the main text focuses on summary-page metric families. One of these
# (the DRAM clock rate) carries the registry's single deferred certificate.
VARIANTS_NOT_PLOTTED = {
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",  # load-only L1 hit rate (outside support)
    "lts__t_requests_srcunit_tex.sum",                         # texture-source request count
    "gpc__cycles_elapsed.avg.per_second",                      # cycle-rate variant
    "sm__cycles_active.avg",                                   # active-cycles variant
    "sm__warps_active.avg.per_cycle_active",                   # occupancy denominator variant
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",   # roll-up
    "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed",  # roll-up
    "dram__cycles_elapsed.avg.per_second",                     # DRAM clock rate (deferred)
}

# Adjudicated in the registry but not promoted to a main metric: under
# non-lockable clocks, wall-clock kernel time is a property of the host's
# frequency governor rather than the kernel.
NOT_PLOTTED_TIME = {
    "gpu__time_duration.sum",  # wall-clock kernel time
}


def _verify_against_artifacts():
    """Assert every hard-coded plotted constant against the raw committed
    artifacts when present (dev machine); skip gracefully otherwise."""
    checks = []

    # --- Map tier counts + the recovered-estimate accounting ---------------
    # The registry adjudicates 20 metrics (4/16); the main-body map presents
    # the 11 essential ones (3/8) with wall-clock kernel time kept out of the
    # plotted map. Assert ALL of it, the per-metric tier of every plotted tile,
    # and that essential + relegated variants + the omitted timing metric
    # exactly partition the registry (so a metric cannot silently vanish).
    m = _load("recoverable_noniden_closure.json")
    if m:
        assert m["n_recoverable"] == 4 and m["n_non_identifiable"] == 16, "registry counts"
        assert m["n_recoverable"] + m["n_non_identifiable"] == 20, "registry total != 20"
        der = m.get("overclaim_guard", {}).get("derivation", {})
        assert der.get("n_recoverable_positive_head") == 2, "2 accepted heads"
        assert der.get("n_certificates_produced") == 15, "15 certificates in registry"
        assert m.get("deferred_metrics") == ["dram__cycles_elapsed.avg.per_second"], "deferred metric"
        tier = {x["metric"]: x["tier"] for x in m["metrics"]}
        plotted = {r[2] for r in CENSUS}
        assert len(plotted) == len(CENSUS) == 11, "essential census is 11 distinct metrics"
        assert not plotted & VARIANTS_NOT_PLOTTED, "a metric is both plotted and relegated"
        assert not plotted & NOT_PLOTTED_TIME, "a metric is both plotted and omitted"
        assert not VARIANTS_NOT_PLOTTED & NOT_PLOTTED_TIME, "a metric is both relegated and omitted"
        assert plotted | VARIANTS_NOT_PLOTTED | NOT_PLOTTED_TIME == set(tier), \
            "essential + variants + omitted timing != registry"
        for _, _, mid, verdict, _, _ in CENSUS:
            want = "recoverable" if verdict == "rec" else "non_identifiable"
            assert tier[mid] == want, f"tier mismatch for {mid}"
        assert tier["gpu__time_duration.sum"] == "non_identifiable", "kernel time tier"
        assert m["deferred_metrics"][0] in VARIANTS_NOT_PLOTTED, "deferred metric is relegated"
        assert sum(1 for r in CENSUS if r[3] == "rec") == 3, "census rec=3"
        assert sum(1 for r in CENSUS if r[4] == "accepted") == 2, "census accepted=2"
        assert sum(1 for r in CENSUS if r[4] == "outside_support") == 1, "census outside-support=1"
        assert sum(1 for r in CENSUS if r[3] == "non") == 8, "census non=8"
        assert all(r[4] == "erased" for r in CENSUS if r[3] == "non"), "all plotted non-id certified"
        checks.append("registry 4/16 over 20; main-body map 3/8 over 11 "
                      "(2 accepted, 1 outside support, 8 certs; 8 variants relegated, "
                      "kernel time omitted from map)")

    # --- Recoverable tier: interconnect traffic estimates ------------------
    dt = _load("sm120_xbar_dual_terminal.json")
    if dt:
        w = dt["heads"][0]["baseline_comparison"]
        r = dt["heads"][1]["baseline_comparison"]
        assert abs(w["head_primary_loss"] - 5.119421114e-07) < 1e-9, "write head loss"
        assert abs(w["baseline_primary_loss"] - 8.027006792e-05) < 1e-7, "write baseline loss"
        assert abs(r["baseline_primary_loss"] - 2.685887918e-12) < 1e-13, "read analytic exact"
        assert w["n_scored_nonzero"] == 39, "write scored n"
        cov = dt["heads"][0]["positive_head_record"]["promotion_evidence"]
        assert abs(cov["coverage_90"]["empirical_coverage"] - 0.8974358974) < 1e-3, "write cov90"
        assert abs(cov["coverage_95"]["empirical_coverage"] - 0.9230769231) < 1e-3, "write cov95"
        assert cov["coverage_90"]["within_band"] and cov["coverage_95"]["within_band"], "write cov in band"
        checks.append("traffic heads: write 157x + coverage in band, read analytic-exact")

    # --- Pre-registered replay grid (printed in the method section) --------
    gp = _load("cache_replay_comparison.json")
    if gp:
        g = gp.get("grid_provenance", {})
        assert g.get("grid_size") == 42, "staged grid is 42 configurations"
        assert g.get("grid_size_cap") == 75, "registered grid cap is 75"
        spread = gp.get("inert_grid_evidence", {}).get("max_grid_prediction_spread")
        assert spread is not None and spread < 1e-3, "grid inert (<1e-3 spread)"
        checks.append("replay grid 42/cap 75, inert (<1e-3 spread)")

    # --- Centrepiece: per-family L2 hit-ratio recovery error ---------------
    v = _load("sm120_l2_selective_verdict.json")
    if v:
        want = {"memory_stress": 0.0415, "normalization": 0.323,
                "other": 0.131, "tensor_compute": 0.920}
        erased = {"other": True, "tensor_compute": True, "register_spill": True,
                  "normalization": False, "memory_stress": False}
        for fam in v["families"]:
            g = fam.get("four_criterion_gate") or {}
            if fam["family"] in want and g.get("full_cohort_median_abs") is not None:
                assert abs(g["full_cohort_median_abs"] - want[fam["family"]]) < 1e-2, fam["family"]
            assert fam.get("is_erased") == erased[fam["family"]], f"is_erased {fam['family']}"
        assert v["corroborating_cross_check"]["agreement_rate"] == 1.0, "cross-check agreement"
        # the normalized x-gate column of tab:l2family (median / 0.055 gate,
        # printed to one decimal in the main-body table)
        med = {f["family"]: (f.get("four_criterion_gate") or {}).get("full_cohort_median_abs")
               for f in v["families"]}
        for fam, x in (("memory_stress", 0.8), ("normalization", 5.9),
                       ("other", 2.4), ("tensor_compute", 16.7)):
            assert abs(med[fam] / 0.055 - x) < 0.06, f"tab:l2family x-gate {fam}"
        checks.append("L2 per-family medians + cross-check + tab:l2family "
                      "x-gate 0.8/5.9/2.4/16.7")
    cl = _load("sm120_cache_metric_closure.json")
    if cl:
        assert abs(cl["named_target_gate"]["threshold_median_abs_error"] - 0.055) < 1e-9, \
            "pre-registered L2 gate 0.055"
        checks.append("L2 pre-registered gate 0.055")

    # --- Timing: clock span + timer agreement + duration predictor ---------
    dp = _load("duration_predictor_report.json")
    if dp:
        dv = dp["dvfs_check"]
        # raw-artifact provenance: the released dvfs_check still includes the
        # deadlock-idle 180 MHz floor (min 180 / span 15.34); the figure and
        # the paper exclude those readings (see DEADLOCK_IDLE_MHZ above).
        assert abs(dv["sm_clock_mhz_min"] - 180.0) < 1 and dv["sm_clock_mhz_max"] > 2700, \
            "raw clock min/max (incl. deadlock-idle floor)"
        assert abs(dv["sm_clock_span_ratio"] - 15.34) < 0.1, \
            "raw clock span (incl. deadlock-idle floor)"
        assert dp["predictor"]["held_out_smape"] > dp["strongest_baseline"]["held_out_smape"], \
            "duration predictor worse than shape baseline (non-identifiable)"
    clk, n_idle = _clock_samples()
    if clk is not None:
        assert n_idle == 6, "deadlock-idle exclusion: exactly the 3 known 180 MHz cells (x2 sources)"
        assert abs(clk.min() - 1687.5) < 1 and clk.max() > 2700, "plotted clock min/max"
        assert abs(clk.max() / clk.min() - 1.64) < 0.01, "plotted clock span 1.64x"
        checks.append("timing clock 1.64x span over 1688-2762 MHz "
                      "(3 deadlock-idle 180 MHz readings excluded) + duration outside support")
    tw = _load("sm120_timing_cupti_ncu_crosswalk.json")
    if tw:
        gaps = [c["cupti_vs_event_rel_gap_median"] for c in tw["cells"]
                if c.get("cupti_vs_event_rel_gap_median") is not None]
        assert max(gaps) < 0.05, "timer agreement within 5%"
        assert abs(max(gaps) * 100 - 2.1) < 0.1, "worst-case timer gap 2.1% (printed in the report)"
        checks.append("two timers agree < 5%")

    # --- Per-surface availability manifest (framework) ---------------------
    SB = os.path.normpath(os.path.join(OUT, "..", "..", "5060_solbench",
                                       "scripts", "baselines_harness"))
    mf = _load(".harness_traces/problem_001_fused_add_rmsnorm_h2048__cuda_cpp/manifest.json", base=SB)
    if mf:
        su = {k: x.get("status") for k, x in mf.get("surfaces", {}).items()}
        assert su.get("nsys") == "unavailable" and su.get("cache_prediction") == "skipped", "surface status"
        assert all(su.get(s) == "ok" for s in ["nvbit_metrics", "cupti_activity", "static", "sanitizer"]), "surface ok"
        checks.append("surface manifest")

    # --- Cross-arch (Ampere, sm_86): the numbers behind the L2 fitted aggregate
    g86 = _load("local_labels/calibration_head_v5_accuracy_gates_sm_86.json", base=_AMP)
    if g86:
        gg = g86["gates"]
        assert abs(gg["full_cohort_geomean_relative_error"] - 0.0642) < 5e-3, "sm_86 L2 geomean 0.064"
        checks.append("Ampere L2 aggregate 0.064 (fitted aggregate)")
    rb = _load("local_labels/calibration_head_v5_robustness_sm_86.json", base=_AMP)
    if rb:
        gg = rb["gates"]
        assert abs(gg["loocv_holdout_relative_gap"] - 0.34) < 0.01, "sm_86 loocv 0.34"
        assert abs(gg["loso_max_relative_change"] - 0.71) < 0.01, "sm_86 loso 0.71"
        checks.append("Ampere robustness 0.34/0.71")
    tp = _load("sm_86_predictor_accuracy_recovery_v2/t2_traffic_prediction_sm_86.json", base=_AMP)
    if tp:
        g = {x["gate"]: x for x in tp["achieved_vs_target"]}
        assert g["write_sectors_p90_rel_error"]["achieved"] < 0.01, "sm_86 write passes"
        assert abs(g["read_sectors_p90_rel_error"]["achieved"] - 0.51) < 0.02, "sm_86 read outside support 0.51"
        checks.append("Ampere recoverable: write passes, read outside support 0.51")

    # --- Microbenchmark validation table (tab:microbench) ------------------
    # Recompute the table's MAPE / exact-cell / Pearson-r constants from the
    # scored per-cell (label, predicted) pairs; p90 gates above come from the
    # artifact's own gate block (rank statistic), not re-derived.
    if tp:
        rd = np.array([(l, p) for _, l, p in _t2_pairs(tp, "read_sectors")])
        wr = np.array([(l, p) for _, l, p in _t2_pairs(tp, "write_sectors")])
        rrel = np.abs(rd[:, 1] - rd[:, 0]) / np.maximum(np.abs(rd[:, 0]), 1e-12)
        wrel = np.abs(wr[:, 1] - wr[:, 0]) / np.maximum(np.abs(wr[:, 0]), 1e-12)
        assert len(rd) == 13 and int((rrel < 1e-9).sum()) == 11, "sm_86 read 11/13 exact"
        assert len(wr) == 20 and int((wrel < 1e-9).sum()) == 19, "sm_86 write 19/20 exact"
        assert abs(rrel.mean() * 100 - 19) < 1.0, "sm_86 read MAPE 19%"
        assert abs(wrel.mean() * 100 - 5) < 0.5, "sm_86 write MAPE 5%"
        assert abs(np.corrcoef(rd[:, 0], rd[:, 1])[0, 1] - 0.94) < 0.01, "sm_86 read r=0.94"
        assert np.corrcoef(wr[:, 0], wr[:, 1])[0, 1] > 0.999, "sm_86 write r>0.999"
        checks.append("microbench sm_86 traffic: 11/13 & 19/20 exact, MAPE 19%/5%, r .94/>.999")
    l1c = _load("local_labels/l1_closure_comparison_sm_86.json", base=_AMP)
    if l1c:
        assert abs(l1c["full_cohort_geomean"] - 0.89) < 0.01, "sm_86 L1 GMRE 0.89"
        assert len(l1c["included_cells"]) == 30, "sm_86 L1 n=30"
        checks.append("microbench sm_86 L1 GMRE 0.89 (n=30)")
    pl = _load("sm120_l2_tree_plru_model.json")
    if pl:
        sc = [c for c in pl["cells"] if c.get("evaluated") and c.get("abs_error") is not None]
        lab = np.array([c["label"] for c in sc])
        prd = np.array([c["predicted"] for c in sc])
        assert len(sc) == 29, "sm_120 L2 scored cohort n=29"
        assert abs(np.corrcoef(lab, prd)[0, 1] - 0.38) < 0.01, "sm_120 L2 cohort r=0.38"
        checks.append("microbench sm_120 L2 cohort r=0.38 (n=29)")
    la = _load("label_admitted.json")
    if la:
        n120 = sum(1 for c in la["cells"] if c.get("arch") == "sm_120")
        assert n120 == 30, "sm_120 admitted cells = 30"
    lg = _load("local_labels/sm86_cohort_regen_ledger.json", base=_AMP)
    if lg:
        n86 = sum(1 for c in lg["cells"] if c.get("status") == "admitted")
        assert n86 == 40, "sm_86 admitted cells = 40"
    if la and lg:
        checks.append("microbench suite 30 (sm_120) / 40 (sm_86) admitted cells")

    # --- Suite accuracy vs the NCU answer key (fig_framework b-d) -----------
    # Every annotation printed in the figure comes from the harness's own
    # registered validation reports; the plotted per-cell points re-apply the
    # frozen deployed calibrator anchors and are summarized + asserted here.
    cells, rep86, rep120, ladder = _suite_data()
    if rep86:
        f, h, m = rep86["cohort_friendly"], rep86["cohort_hostile"], rep86["meta"]
        assert f["cell_count"] == 34 and h["cell_count"] == 1, "sm_86 cohort sizes"
        assert abs(f["l1_mape"] - 0.0329) < 5e-4 and abs(f["l2_mape"] - 0.0549) < 5e-4, \
            "sm_86 LOOCV cohort errors 0.033/0.055"
        assert f["l1_gate_threshold"] == 0.05 and f["l2_gate_threshold"] == 0.06, "sm_86 gates"
        assert abs(h["l2_mape"] - 0.0317) < 5e-4 and h["l1_mape"] == 0.0, "sm_86 hostile errors"
        assert rep86["overall"]["all_gates_pass"], "sm_86 suite gates pass"
        assert m["admitted_cell_count"] == 35 and m["excluded_cell_count"] == 30, \
            "sm_86 admitted/typed-excluded 35/30"
        checks.append("suite sm_86: LOOCV 0.033/0.055 (n=34) + hostile, gates pass, 30 typed-excluded")
    if rep120:
        f, h = rep120["cohort_friendly"], rep120["cohort_hostile"]
        assert f["cell_count"] == 10 and h["cell_count"] == 2, "sm_120 cohort sizes"
        assert abs(f["l1_mape"] - 0.0357) < 5e-4 and abs(f["l2_mape"] - 0.0378) < 5e-4, \
            "sm_120 held-out cohort errors 0.036/0.038"
        assert h["l1_mape"] == 0.0 and abs(h["l2_mape"] - 0.0203) < 5e-4, "sm_120 hostile errors"
        assert rep120["overall"]["all_gates_pass"], "sm_120 suite gates pass"
        checks.append("suite sm_120: held-out 0.036/0.038 (n=10) + hostile (n=2), gates pass")
    if cells:
        assert len(cells) == 35, "full admitted cohort drawn (34 friendly + 1 hostile)"
        from collections import Counter as _Ctr
        assert dict(_Ctr(c["genre"] for c in cells)) == {"L1": 18, "FIB": 15, "L2": 2}, \
            "plotted suite genre census"
        e1 = np.array([abs(c["cal_l1"] - c["ncu_l1"]) for c in cells])
        e2 = np.array([abs(c["cal_l2"] - c["ncu_l2"]) for c in cells])
        assert abs(e1.mean() - 0.0094) < 1e-3 and abs(e2.mean() - 0.0435) < 1e-3, \
            "deployed-calibrator point cloud mean |err| (in-sample)"
        assert e1.max() < 0.08 and e2.max() < 0.15, "deployed-calibrator max |err|"
        # the panel-(b) origin cluster and the panel-(c) fitted plateau, both
        # printed in the figure/caption
        n00 = sum(1 for c in cells if c["ncu_l1"] < 0.005 and c["cal_l1"] < 0.005)
        npl = sum(1 for c in cells if abs(c["cal_l2"] - _L2_REUSE_PLATEAU) < 1e-9)
        assert n00 == 17, "17 zero-truth/zero-pred L1 cells"
        assert npl == 15, "15 cells on the fitted L2 reuse plateau 0.91"
        # the typed-exclusion split printed in §V (15 capability / 15 evidence)
        cls = _Ctr(e["classification"] for e in rep86["per_cell"])
        assert cls == {"measured": 35, "capability_excluded": 15,
                       "raw_unavailable_excluded": 8, "ncu_terminal_excluded": 7}, \
            "typed-exclusion classification split"
        # anchors round-trip: the stored frozen calibrator anchors reproduce
        # every stored deployed L1 prediction exactly (np.interp == sklearn
        # IsotonicRegression linear interpolation with clip)
        iso = _load_toml("isotonic_params_sm86_expanded.toml", _SB86)
        if iso:
            l1x, l1y = iso["l1_head"]["x_thresholds"], iso["l1_head"]["y_thresholds"]
            pcm = {e["slug"]: e for e in rep86["per_cell"]}
            assert all(abs(np.interp(pcm[c["slug"]]["predicted_l1_raw"], l1x, l1y)
                           - c["cal_l1"]) < 1e-9 for c in cells), "L1 anchor round-trip"
            assert abs(iso["l2_head"]["reuse"]["y_thresholds"][0] - _L2_REUSE_PLATEAU) < 1e-12 \
                and iso["l2_head"]["reuse"]["y_thresholds"][0] == iso["l2_head"]["reuse"]["y_thresholds"][-1], \
                "L2 reuse branch is a fitted constant (the disclosed plateau)"
        checks.append("suite point cloud: n=35 (L1 18 / FIB 15 / L2 2), 17 at origin, "
                      "15 on the fitted 0.91 plateau, exclusions 15/8/7, anchors round-trip")
    if ladder:
        assert sum(ladder.values()) == 235, "ladder covers all 235 problems"
        assert ladder.get("mape_scored") == 12, "sm_120 ladder: 12 scored"
        assert ladder.get("predictor_prediction_available") == 51, "sm_120 ladder: 51 trace-ready"
        checks.append("sm_120 readiness ladder 12 scored / 51 trace-ready of 235")

    # --- Microbench per-cell scatters (fig_microbench) ----------------------
    tr = _sm120_traffic_rows()
    if tr:
        lab = np.array([l for l, _ in tr]); prd = np.array([p for _, p in tr])
        rel = np.abs(prd - lab) / np.maximum(lab, 1e-12)
        assert len(tr) == 158, "sm_120 read cohort n=158"
        assert int((rel < 1e-9).sum()) == 136, "sm_120 raw read exact on 136/158"
        assert np.percentile(rel, 90) < 1e-4 and rel.max() < 0.07, "sm_120 raw read tail"
        checks.append("sm_120 raw read recovery exact 136/158, p90 < 1e-4")
    if tp:
        worst = sorted(_t2_pairs(tp, "read_sectors"),
                       key=lambda t: -abs(t[2] - t[1]) / t[1])[:2]
        assert worst[0][0] == "l2_persisting_window_sweep" and \
            abs(abs(worst[0][2] - worst[0][1]) / worst[0][1] - 1.93) < 0.01, \
            "sm_86 read outlier #1: persisting-window sweep, rel err 1.93"
        assert worst[1][0] == "rmsnorm_fwd_bf16" and \
            abs(abs(worst[1][2] - worst[1][1]) / worst[1][1] - 0.51) < 0.01, \
            "sm_86 read outlier #2: full-line fetch, rel err 0.51"
        checks.append("sm_86 read outliers 1.93/0.51 (named cells)")
    l1cells = _sm86_l1_cells()
    if l1cells:
        truth = np.array([t for t, _ in l1cells])
        pred = np.array([p for _, p in l1cells])
        assert len(l1cells) == 30 and int((truth < 0.02).sum()) == 12, \
            "sm_86 L1 cohort: 30 cells, 12 bypass-class (true rate ~0)"
        nb = truth >= 0.02
        assert abs(np.median(np.abs(pred[nb] - truth[nb])) - 0.134) < 5e-3, \
            "sm_86 L1 non-bypass median |err| 0.13 (printed in the caption)"
        checks.append("sm_86 L1 scatter: 12/30 bypass-class cells, non-bypass median 0.13")
    l2c = _sm120_l2_cells()
    if l2c:
        from collections import Counter as _Ctr
        fam = _Ctr(f for f, _, _ in l2c)
        assert fam == {"tensor_compute": 9, "other": 7, "activation_norm": 7,
                       "memory_stress": 6}, "sm_120 L2 family census"
        # two low-locality memory-streaming cells collapse to zero prediction
        # (named in the caption so the recovered-family claim stays per-cell
        # honest); the family median 0.0415 is asserted above
        ms0 = sum(1 for f, l, p in l2c if f == "memory_stress" and p < 0.005)
        assert ms0 == 2, "2 memory-streaming cells at zero prediction"
        checks.append("sm_120 L2 scatter family census 6/7/7/9 (2 streaming cells collapsed)")

    if checks:
        print("artifact-verified:", "; ".join(checks))
    else:
        print("artifact check skipped (raw artifacts not present)")


def _panel_tag(ax, s):
    """Bold (a)/(b) tag in the top-left corner inside a panel, matching the
    (a)/(b) references in the LaTeX captions."""
    ax.text(0.012, 0.975, s, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, weight="bold", color="black", zorder=10)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


# ===========================================================================
# FIG 3 -- The identifiability map as a visual census of the 11 essential
#   metrics (wall-clock kernel time is not plotted -- see NOT_PLOTTED_TIME).
#   Each metric is a tile, grouped by what it measures, coloured by
#   verdict. The point of the picture: recoverability lives ONLY at the
#   L1/interconnect boundary (left); everything global, temporal, or
#   throughput-related is lost.
# ===========================================================================
def fig_map():
    REC, NON = TIER["recoverable"], TIER["non_identifiable"]
    # group order: recoverable-bearing groups first, then the "all lost" sea
    groups = ["Cache traffic\n(L1↔L2)", "Cache hit\nrate", "DRAM\ntraffic",
              "Cycles", "Occupancy", "Throughput\n& IPC"]
    col = {g: i for i, g in enumerate(groups)}
    stacks = {g: [] for g in groups}
    for grp, label, _mid, verdict, status, headline in CENSUS:
        stacks[grp].append((label, verdict, status, headline))

    fig, ax = plt.subplots(figsize=(ps.DBL_W, 2.24))
    tw, th = 0.92, 0.86       # tile width/height
    maxrows = max(len(stacks[g]) for g in groups)

    for g in groups:
        x = col[g]
        for r, (label, verdict, status, headline) in enumerate(stacks[g]):
            base = REC if verdict == "rec" else NON
            light = "#cfe3f2" if verdict == "rec" else "#f8ddcb"
            if status == "accepted":
                fc, ec, hatch, txtcol, wt = base, "black", None, "white", "bold"
            elif status == "outside_support":
                fc, ec, hatch, txtcol, wt = light, base, "////", base, "bold"
            else:  # erased
                fc, ec, hatch, txtcol, wt = base, "black", None, "white", "normal"
            box = FancyBboxPatch((x - tw / 2, r + (1 - th) / 2), tw, th,
                                 boxstyle="round,pad=0.012,rounding_size=0.08",
                                 linewidth=2.0 if headline else 0.8,
                                 edgecolor="black" if headline else ec,
                                 facecolor=fc, hatch=hatch, zorder=2)
            ax.add_patch(box)
            # outside-support tile: one solid white chip behind the whole label (a
            # per-glyph halo lets hatch strokes peek through inter-word gaps)
            chip = (dict(boxstyle="round,pad=0.18", facecolor="white",
                         edgecolor="none")
                    if status == "outside_support" else None)
            ax.text(x, r + (0.58 if headline else 0.50), label, ha="center",
                    va="center", fontsize=6.5, color=txtcol, zorder=3, weight=wt,
                    bbox=chip)
            if status == "accepted":
                ax.scatter([x + tw / 2 - 0.13], [r + th - 0.10], marker="*", s=34,
                           color="white", edgecolors="none", zorder=4)
            if headline:
                # headline flag: thick black border (drawn above) + italic word.
                # No star here -- the star glyph is reserved for tiles whose
                # estimate was accepted, and the headline tile is erased.
                ax.text(x, r + 0.215, "headline", ha="center", va="center",
                        fontsize=6.5, style="italic", color="white", zorder=3)

    # divider between the recoverable-bearing cache groups and the lost "sea"
    ax.axvline(1.5, color="#555555", lw=1.0, ls=(0, (4, 3)), zorder=1)
    # each side header sits just above its own column group (left stacks are
    # 3 tall, the lost "sea" is up to 4 tall)
    left_top = max(len(stacks[g]) for g in groups[:2])
    ax.text(0.5, left_top + 0.18, "recoverable verdicts occur only here:\nL1 / interconnect boundary",
            ha="center", va="bottom", fontsize=7.0, color=REC, weight="bold")
    ax.text(3.5, maxrows + 0.18, "lost side:\nglobal, temporal & throughput metrics",
            ha="center", va="bottom", fontsize=7.0, color=NON, weight="bold")

    for g in groups:
        ax.text(col[g], -0.30, g, ha="center", va="top", fontsize=6.5,
                color="#222222")

    ax.set_xlim(-0.65, len(groups) - 0.35)
    ax.set_ylim(-0.88, maxrows + 0.95)
    ax.axis("off")
    legend = [
        Patch(facecolor=REC, edgecolor="black", label="recoverable — estimate accepted (2)"),
        Patch(facecolor="#cfe3f2", edgecolor=REC, hatch="////", label="recoverable — outside support (1)"),
        Patch(facecolor=NON, edgecolor="black", label="non-identifiable — missing signal named (8)"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.19),
              ncol=3, fontsize=6.5, handlelength=1.5, columnspacing=1.4,
              borderpad=0.4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.13)
    save(fig, "fig_map.pdf")


# ===========================================================================
# FIG (cut) -- The recoverable tier, shown as a positive result with calibrated
#   uncertainty. (a) the two accepted interconnect-traffic estimates beat the
#   strongest analytic baseline (write by 157x; read reproduced exactly).
#   (b) the calibrated write estimate's error bars pass their coverage check.
#   source: sm120_xbar_dual_terminal.json (heads[0]=write, heads[1]=read).
# ===========================================================================
def fig_recoverable():
    REC = TIER["recoverable"]
    dt = _load("sm120_xbar_dual_terminal.json")
    if dt:
        w = dt["heads"][0]["baseline_comparison"]
        r = dt["heads"][1]["baseline_comparison"]
        write_head, write_base = w["head_primary_loss"], w["baseline_primary_loss"]
        read_head, read_base = r["head_primary_loss"], r["baseline_primary_loss"]
        pev = dt["heads"][0]["positive_head_record"]["promotion_evidence"]
        cov = {0.90: pev["coverage_90"], 0.95: pev["coverage_95"]}
    else:  # frozen fallback
        write_head, write_base = 5.119421114e-07, 8.027006792e-05
        read_head, read_base = 2.894808801e-10, 2.685887918e-12
        cov = {0.90: {"empirical_coverage": 0.8974358974, "binomial_band_low": 0.7948717949,
                      "binomial_band_high": 0.9743589744, "n_covered": 35, "n": 39},
               0.95: {"empirical_coverage": 0.9230769231, "binomial_band_low": 0.8974358974,
                      "binomial_band_high": 1.0, "n_covered": 36, "n": 39}}

    # single-column, stacked panels (the report's 8-page budget)
    fig, (a, b) = plt.subplots(2, 1, figsize=(ps.COL_W, 3.55),
                               gridspec_kw={"hspace": 0.62})

    # (a) accepted recovery vs strongest baseline, geometric-mean relative
    #   error, as a dot/interval plot on a log axis (no bars from a 1e-12 floor).
    #   write: a CALIBRATED estimate beats the analytic baseline by 157x.
    #   read : the analytic reuse-distance baseline reproduces it EXACTLY, so
    #          the accepted recovery and the baseline coincide (one filled dot
    #          -- the same legend glyph as the accepted recovery; the
    #          annotation states the coincidence).
    y_write, y_read = 0.62, 0.0
    a.grid(False)
    a.grid(axis="x", ls=":", lw=0.5, alpha=0.35, color="#666666")
    a.plot([write_head, write_base], [y_write, y_write], color="#888888",
           lw=0.9, zorder=2)
    a.plot([write_base], [y_write], marker="o", ms=7, mfc="white",
           mec="#666666", mew=1.1, ls="none", zorder=3,
           label="strongest analytic baseline")
    a.plot([write_head], [y_write], marker="o", ms=7, mfc=REC, mec="black",
           mew=0.6, ls="none", zorder=4, label="accepted counter-free recovery")
    a.plot([read_base], [y_read], marker="o", ms=7, mfc=REC, mec="black",
           mew=0.6, ls="none", zorder=4)
    a.set_xscale("log")
    a.set_xlim(4e-13, 6e-3)
    a.set_yticks([y_read, y_write])
    a.set_yticklabels(["L1→L2 read", "L1→L2 write\n(calibrated)"], fontsize=7)
    a.set_ylim(-0.42, 1.30)
    a.set_xlabel("geometric-mean relative error (lower is better)", fontsize=7.5)
    a.annotate("157× better", (np.sqrt(write_head * write_base), y_write + 0.10),
               ha="center", va="bottom", fontsize=6.8, color=REC, weight="bold")
    a.annotate("reproduced exactly by the analytic\nbaseline — the two coincide",
               (read_base, y_read), xytext=(read_base * 25, y_read - 0.02),
               fontsize=6.5, color=REC, weight="bold", va="center", ha="left")
    a.legend(loc="upper left", bbox_to_anchor=(0.03, 1.04), fontsize=6.2,
             handletextpad=0.5, borderaxespad=0.2)
    a.tick_params(axis="x", labelsize=6.5)
    _panel_tag(a, "(a)")

    # (b) calibrated write estimate: empirical coverage falls inside the band
    targets = [0.90, 0.95]
    emp = [cov[t]["empirical_coverage"] for t in targets]
    x = np.arange(len(targets))
    # acceptance band (binomial) drawn as a vertical span per target
    for xi, t in zip(x, targets):
        b.add_patch(Rectangle((xi - 0.28, cov[t]["binomial_band_low"]), 0.56,
                              cov[t]["binomial_band_high"] - cov[t]["binomial_band_low"],
                              facecolor=REC, alpha=0.16, edgecolor="none", zorder=1))
    b.plot(x, targets, marker="_", ms=22, mew=1.6, ls="none", color="#555555",
           label="nominal (target) coverage", zorder=2)
    b.plot(x, emp, marker="o", ms=8, ls="none", color=REC, markeredgecolor="black",
           markeredgewidth=0.6, label="empirical coverage", zorder=3)
    for xi, t in zip(x, targets):
        b.annotate(f"{cov[t]['empirical_coverage']*100:.0f}%\n({cov[t]['n_covered']}/{cov[t]['n']})",
                   (xi, cov[t]["empirical_coverage"]), xytext=(xi + 0.34, cov[t]["empirical_coverage"]),
                   fontsize=6.5, va="center", color=REC, weight="bold")
    b.set_xticks(x)
    b.set_xticklabels(["90% band", "95% band"])
    b.set_ylabel("coverage of the calibrated\nerror bar", fontsize=7.5)
    b.set_ylim(0.70, 1.02)
    b.set_xlim(-0.65, 1.85)
    # framed legend so its glyphs cannot be mistaken for data points
    b.legend(loc="lower right", fontsize=6.5, frameon=True, facecolor="white",
             framealpha=0.95, edgecolor="#dddddd")
    _panel_tag(b, "(b)")
    save(fig, "fig_recoverable.pdf")


# ===========================================================================
# FIG 4 -- CENTREPIECE: large-cache (L2) hit-ratio recovery error by workload
#   family. Low error for streaming kernels; collapses once reuse moves on-chip.
#   source: sm120_l2_selective_verdict.json (per-family median absolute error).
# ===========================================================================
def fig_l2_family():
    REC, NON = TIER["recoverable"], TIER["non_identifiable"]
    # short two-line tick labels: no glyph contact at column width
    fams = ["memory\nstreaming", "reduction /\nnormalize", "intermediate\nbuffers",
            "matrix\nmultiply", "register\nspill"]
    med = [0.0415, 0.323, 0.131, 0.920, None]   # None = no observable cell (structural)
    nfam = [6, 7, 7, 9, 0]
    # verdict per family: recoverable / gate-failed-with-mechanism / erased
    kind = ["recoverable", "above-threshold", "not-identifiable", "not-identifiable", "not-identifiable"]
    mech = ["recoverable\n(n=6)", "cross-group\nreuse unseen\n(n=7)",
            "staging buffers\nunseen (n=7)", "operand tiles staged on-chip (n=9)",
            None]
    colmap = {"recoverable": REC, "above-threshold": C["orange"], "not-identifiable": NON}

    # single column (the report's 8-page budget): 5 bars read fine at 3.35 in
    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.38))
    x = np.arange(len(fams))
    w = 0.62
    for i, (m, k) in enumerate(zip(med, kind)):
        if m is None:
            # register-spill yields NO scorable cell (reuse is thread-private):
            # an x marker only -- deliberately NOT a bar, so it cannot be read
            # as a passing low-error recovery below the gate line
            ax.scatter([i], [0.012], marker="x", s=52, color=NON, linewidths=1.8,
                       zorder=4, clip_on=False)
            ax.annotate("", (i, 0.04), (i, 0.27),
                        arrowprops=dict(arrowstyle="-", color=NON, lw=0.7))
            ax.annotate("no observable\ncell (n=0):\nreuse is\nthread-private",
                        (i + 0.02, 0.295), ha="center", va="bottom",
                        fontsize=6.5, color=NON)
            continue
        ax.bar(i, m, w, color=colmap[k], edgecolor="black", linewidth=0.7, zorder=2,
               hatch="\\\\" if k == "above-threshold" else None)
        if m > 0.55:   # tall bar: value above the bar, mechanism rotated inside
            ax.annotate(f"{m:.2f}", (i, m + 0.016), ha="center", va="bottom",
                        fontsize=9, weight="bold", color="#111111")
            ax.text(i, m / 2, mech[i], rotation=90, ha="center", va="center",
                    fontsize=6.5, color="white")
        else:          # short bar: value bold above the bar, mechanism above it
            ax.annotate(f"{m:.2f}", (i, m + 0.018), ha="center", va="bottom",
                        fontsize=9, weight="bold", color="#111111")
            ax.annotate(mech[i], (i, m + 0.115), ha="center",
                        va="bottom", fontsize=6.5, color="#333333")

    ax.axhline(0.055, color=C["green"], ls=(0, (5, 2)), lw=1.4, zorder=1)
    # gate label parked in the empty band between the reduction and
    # intermediate-buffer bars, with a thin leader down to the line itself
    ax.annotate("fixed\nthreshold 0.055", xy=(-0.45, 0.062), xytext=(-0.60, 0.44),
                fontsize=6.5, color=C["green"], ha="left", va="bottom",
                weight="bold",
                arrowprops=dict(arrowstyle="-", color=C["green"], lw=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(fams, fontsize=6.6)
    ax.set_ylabel("error in recovered L2 hit ratio\n(median |predicted $-$ true|)",
                  fontsize=7.5)
    ax.set_ylim(0, 1.06)
    ax.set_xlim(-0.7, 4.7)

    handles = [Patch(facecolor=REC, edgecolor="black", label="recoverable"),
               Patch(facecolor=C["orange"], edgecolor="black", hatch="\\\\",
                     label="above threshold $\\Rightarrow$ not identifiable"),
               Patch(facecolor=NON, edgecolor="black", label="not identifiable")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, ncol=1,
              bbox_to_anchor=(0.005, 0.995), framealpha=0.95, frameon=True,
              edgecolor="#dddddd", borderpad=0.4, labelspacing=0.4)
    save(fig, "fig_l2_family.pdf")


# ===========================================================================
# FIG 5 -- Counter-free timing: the SM clock moves across a 1.6x range under
#   DVFS between admissible runs and cannot be locked, so absolute time is
#   only an interval. The deadlock-idle 180 MHz readings are excluded (see
#   DEADLOCK_IDLE_MHZ): the GPU was wedged, not computing.
#   source: sm120_timing_cupti_ncu_crosswalk.json, duration_predictor_report.json.
# ===========================================================================
def fig_timing():
    G5060, G3090 = ps.GPU_COLOR["rtx5060"], ps.GPU_COLOR["rtx3090ti"]
    # single panel, single column (the report's 8-page budget): the clock-swing
    # evidence; the timer-agreement numbers (worst case 2.1%, <5% tolerance,
    # also <5% on Ampere) are stated in the text and asserted in _verify above.
    fig, b = plt.subplots(figsize=(ps.COL_W, 1.66))

    # observed SM clock states under DVFS -> time is an interval, not a point
    clocks, _ = _clock_samples()
    if clocks is None:  # frozen fallback (deadlock-idle readings already excluded)
        clocks = np.array([1687.5, 1973.0, 2130.0, 2130.0, 2750.9, 2761.7])
    cmin, cmax = clocks.min(), clocks.max()
    cmed = float(np.median(clocks))
    span = cmax / cmin
    # 5060 swarm on a number line (y=1), Ampere range as a band (y=0.25);
    # GPU identity uses the neutral teal/slate pair, never the verdict colours
    rng = np.random.RandomState(7)
    jit = (rng.rand(len(clocks)) - 0.5) * 0.16
    b.grid(False)
    b.grid(axis="x", which="major", ls=":", lw=0.5, alpha=0.35, color="#666666")
    b.scatter(clocks, np.full(len(clocks), 1.0) + jit, s=26, color=G5060,
              edgecolor="black", linewidth=0.4, zorder=3, alpha=0.85)
    b.plot([cmin, cmax], [1.0, 1.0], color=C["grey"], lw=1.0, zorder=1)
    b.scatter([cmed], [1.0], marker="D", s=30, color="#13554c", edgecolor="black",
              linewidth=0.5, zorder=5)  # distinguish the median
    for val, lab, dy in [(cmin, f"min {cmin:.0f}", -0.26), (cmed, f"median {cmed:.0f}", 0.30),
                         (cmax, f"max {cmax:.0f}", -0.26)]:
        b.annotate(lab, (val, 1.0 + dy), ha="center",
                   va="bottom" if dy > 0 else "top", fontsize=6.5, color="#13554c")
    b.annotate(f"{span:.1f}$\\times$ swing — not lockable without root",
               ((cmin + cmax) / 2, 1.56), ha="center", fontsize=6.8,
               color="#5B6770", weight="bold")
    # Ampere observed band 1350-2130 MHz
    b.plot([1350, 2130], [0.25, 0.25], color=G3090, lw=5, alpha=0.65,
           solid_capstyle="round", zorder=2)
    b.annotate("1350–2130 MHz observed,\nnot lockable either", (1740, 0.02),
               ha="center", va="top", fontsize=6.5, color=G3090)
    b.set_xlim(1230, 2980)
    b.set_xticks([1400, 1700, 2000, 2300, 2600, 2900])
    b.set_ylim(-0.45, 2.0)
    b.set_yticks([0.25, 1.0])
    b.set_yticklabels(["RTX 3090 Ti\n(Ampere)", "RTX 5060\n(Blackwell)"],
                      fontsize=6.5)
    b.set_xlabel("observed SM clock (MHz)", fontsize=7.5)
    save(fig, "fig_timing.pdf")


# ===========================================================================
# FIG -- The framework at benchmark scale, including measured accuracy
#   against the NCU answer key:
#   (a) per-genre availability on both cards;
#   (b,c) per-problem deployed-calibrator L1 / L2 read-sector hit-rate
#         predictions vs the NCU answer key, coloured by genre (the FULL
#         sm_86 admitted cohort, all 35 cells from the validation report's
#         [[per_cell]] record); the quoted
#         cohort errors/gates are the harness's registered LOOCV numbers;
#   (d) the per-cohort gate outcomes on BOTH cards.
# ===========================================================================
def fig_framework():
    G5060, G3090 = ps.GPU_COLOR["rtx5060"], ps.GPU_COLOR["rtx3090ti"]
    cells, rep86, rep120, ladder = _suite_data()
    fig = plt.figure(figsize=(ps.DBL_W, 2.42))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.08, 0.92, 0.92, 1.5],
                          wspace=0.42, left=0.055, right=0.995,
                          top=0.90, bottom=0.165)
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1])
    c = fig.add_subplot(gs[2])
    d = fig.add_subplot(gs[3])

    # ---- (a) coverage per genre, both cards (unchanged content) -----------
    cats = ["L1", "L2", "Quant", "FIB"]
    suite = [94, 82, 33, 26]            # published SOL-ExecBench sizes (sum 235)
    avail_5060 = [90, 75, 31, 26]       # sm_120 (8 GiB): 222 / 235
    avail_3090 = [92, 75, 32, 26]       # sm_86  (24 GiB): 225 / 235
    x = np.arange(len(cats)); bw = 0.38
    a.bar(x, suite, bw * 2 + 0.12, facecolor="#ededed", edgecolor=C["grey"],
          linewidth=0.7, label="in suite (235)", zorder=1)
    a.bar(x - bw / 2, avail_5060, bw, color=G5060, edgecolor="black",
          linewidth=0.6, label="RTX 5060 (8 GiB)", zorder=2)
    a.bar(x + bw / 2, avail_3090, bw, color=G3090, edgecolor="black",
          linewidth=0.6, hatch="///", label="RTX 3090 Ti (24 GiB)", zorder=2)
    for xi in range(len(cats)):
        a.annotate(str(suite[xi]), (xi, suite[xi] + 2.0), ha="center",
                   fontsize=6.5, color="#777777")
        for dx, val, dark in ((-bw / 2, avail_5060[xi], "#13554c"),
                              (bw / 2, avail_3090[xi], "#2b3f52")):
            a.annotate(str(val), (xi + dx, val - 2.5), ha="center", va="top",
                       fontsize=6.0, color="white", weight="bold", zorder=4,
                       path_effects=[pe.withStroke(linewidth=1.5, foreground=dark)])
    a.set_xticks(x); a.set_xticklabels(cats, fontsize=7)
    a.set_ylabel("kernel problems", fontsize=7)
    a.set_ylim(0, 119)
    a.legend(loc="upper right", fontsize=5.8, handlelength=1.2, borderpad=0.25,
             labelspacing=0.35, handletextpad=0.4)
    _panel_tag(a, "(a)")

    # ---- (b,c) per-problem prediction vs NCU answer key (sm_86) -----------
    # Drawn straight from the validation report's per-cell record of the
    # deployed calibrated predictions (full admitted cohort, n=35). Genre is
    # double-encoded (colour + marker shape) so it survives grayscale and
    # cannot be confused with the GPU teal/slate of panels (a,d).
    if cells is None:
        cells = []
    f86 = rep86["cohort_friendly"] if rep86 else {"l1_mape": 0.0329, "l2_mape": 0.0549,
                                                  "cell_count": 34}
    GMARK = {"L1": "o", "L2": "s", "FIB": "^"}
    for ax, key_t, key_p, band, mape, gate, tag, apos in (
            (b, "ncu_l1", "cal_l1", 0.05, f86["l1_mape"], 0.05, "(b)",
             dict(xy=(0.96, 0.05), ha="right", va="bottom")),
            (c, "ncu_l2", "cal_l2", 0.06, f86["l2_mape"], 0.06, "(c)",
             dict(xy=(0.07, 0.88), ha="left", va="top"))):
        ax.fill_between([0, 1], [-band, 1 - band], [band, 1 + band],
                        color="#000000", alpha=0.07, lw=0, zorder=1)
        ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, zorder=2)
        for g in ("L1", "L2", "FIB"):   # no Quant cell is NCU-labelable yet
            xs = [cc[key_t] for cc in cells if cc["genre"] == g]
            ys = [cc[key_p] for cc in cells if cc["genre"] == g]
            ax.scatter(xs, ys, s=13, color=GENRE_COLOR[g], edgecolor="black",
                       linewidth=0.35, zorder=3, marker=GMARK[g],
                       label={"L1": "L1 genre", "L2": "L2 genre",
                              "FIB": "FIB genre"}[g])
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("NCU reference (fraction)", fontsize=6.8, labelpad=1.5)
        ax.annotate(f"mean |err| {mape:.3f} $\\leq$ {gate:.2f} $\\checkmark$\n"
                    f"(leave-one-out, $n{{=}}{f86['cell_count']}$)",
                    apos["xy"], xycoords="axes fraction", ha=apos["ha"],
                    va=apos["va"], fontsize=6.0, color="#111111")
        _panel_tag(ax, tag)
    n00 = sum(1 for cc in cells if cc["ncu_l1"] < 0.005 and cc["cal_l1"] < 0.005)
    if n00:
        b.annotate(f"{n00} cells at (0,\\,0): bypass/streaming\nkernels, "
                   "correctly predicted zero",
                   xy=(0.01, 0.012), xytext=(0.33, 0.235), xycoords="data",
                   textcoords="axes fraction", fontsize=5.6, color="#444444",
                   ha="left", va="bottom",
                   arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6,
                                   shrinkB=3))
    npl = sum(1 for cc in cells if abs(cc["cal_l2"] - _L2_REUSE_PLATEAU) < 1e-9)
    if npl:
        c.annotate(f"{npl} staged-reuse cells share one\nfitted value (0.91): "
                   "a family-level estimate,\nnot a per-kernel recovery",
                   (0.96, 0.30), xycoords="axes fraction", ha="right",
                   va="top", fontsize=5.6, color="#444444")
    b.set_ylabel("predicted hit rate", fontsize=6.8, labelpad=1.5)
    b.set_title("L1 load-sector hit rate (calibrated)", fontsize=6.8, pad=2.5)
    c.set_title("L2 read-sector hit rate (calibrated)", fontsize=6.8, pad=2.5)
    b.legend(loc="upper left", fontsize=5.6, handletextpad=0.15,
             borderpad=0.25, labelspacing=0.3, bbox_to_anchor=(0.005, 0.86))

    # ---- (d) cohort gates on both cards -----------------------------------
    r86f = rep86["cohort_friendly"] if rep86 else {"l1_mape": 0.0329, "l2_mape": 0.0549}
    r86h = rep86["cohort_hostile"] if rep86 else {"l1_mape": 0.0, "l2_mape": 0.0317}
    r120f = rep120["cohort_friendly"] if rep120 else {"l1_mape": 0.0357, "l2_mape": 0.0378}
    r120h = rep120["cohort_hostile"] if rep120 else {"l1_mape": 0.0, "l2_mape": 0.0203}
    rows = [  # (label, colour, L1 err, L2 err)
        ("3090 Ti main set ($n{=}34$)", G3090, r86f["l1_mape"], r86f["l2_mape"]),
        ("3090 Ti stress set ($n{=}1$)",   G3090, r86h["l1_mape"], r86h["l2_mape"]),
        ("5060 held-out ($n{=}10$)",    G5060, r120f["l1_mape"], r120f["l2_mape"]),
        ("5060 stress set ($n{=}2$)",      G5060, r120h["l1_mape"], r120h["l2_mape"]),
    ]
    yy = [4.85, 4.00, 3.15, 2.30]
    d.grid(False)
    d.grid(axis="x", ls=":", lw=0.5, alpha=0.35, color="#666666")
    for gx, lab, ha in ((0.05, "thresholds 0.05 /", "right"),
                        (0.06, "0.06\n(3090 Ti L2)", "left"),
                        (0.15, "stress 0.15", "center")):
        d.axvline(gx, color=C["green"], ls=(0, (4, 2)), lw=0.9, zorder=1)
        d.annotate(lab, (gx, 5.62), ha=ha, va="bottom", fontsize=5.6,
                   color=C["green"], weight="bold")
    chip = dict(boxstyle="round,pad=0.13", facecolor="white", edgecolor="none")
    for (lab, col, e1, e2), y in zip(rows, yy):
        d.scatter([e1], [y + 0.17], s=22, color=col, edgecolor="black",
                  linewidth=0.4, zorder=4, marker="o")
        d.scatter([e2], [y - 0.17], s=24, color=col, edgecolor="black",
                  linewidth=0.4, zorder=4, marker="D")
        d.annotate(lab, (0.066, y), fontsize=5.8, color="#222222",
                   ha="left", va="center", zorder=3, bbox=chip)
    # mark the L1-circle / L2-diamond convention once, on the first row
    d.annotate("L1", (rows[0][2] + 0.006, yy[0] + 0.17), fontsize=5.4,
               color="#222222", ha="left", va="center", zorder=4)
    d.annotate("L2", (rows[0][3] + 0.006, yy[0] - 0.17), fontsize=5.4,
               color="#222222", ha="left", va="center", zorder=4)
    d.set_yticks([])
    d.set_xlim(-0.008, 0.168)
    d.set_xticks([0, 0.05, 0.10, 0.15])
    d.set_ylim(1.85, 5.6)
    d.tick_params(labelsize=6.5)
    d.set_xlabel("cohort mean |pred $-$ truth| (fraction)",
                 fontsize=6.6, labelpad=1.5)
    _panel_tag(d, "(d)")
    save(fig, "fig_framework.pdf")


# ===========================================================================
# FIG -- Microbenchmark validation as per-cell scatter against the NCU answer
#   key:
#   the evidence behind the microbench table (Table V) and the l2family
#   centerpiece (Table II), one panel per metric dimension. Accepted estimates sit
#   on the diagonal; outside-support/erased ones show their diagnosed failure shape
#   instead of being averaged away.
#   sources: sm120_xbar_head_cohort_diagnostic.json (read traffic, full
#   158-cell cohort), t2_traffic_prediction_sm_86.json (13 read / 20 write),
#   l1_closure_comparison_sm_86.json (30 cells), sm120_l2_tree_plru_model.json
#   (29 scored cells, family-tagged).
# ===========================================================================
def fig_microbench():
    REC, NON = TIER["recoverable"], TIER["non_identifiable"]
    fig = plt.figure(figsize=(ps.DBL_W, 2.46))
    gs = fig.add_gridspec(1, 4, wspace=0.40, left=0.052, right=0.995,
                          top=0.885, bottom=0.17)
    a, b, c, d = [fig.add_subplot(gs[i]) for i in range(4)]

    # ---- (a) read sectors, sm_120: the exact leg --------------------------
    tr = _sm120_traffic_rows()
    if tr is None:
        tr = [(2.56e3, 2.56e3), (3.58e4, 3.58e4)]  # layout fallback (stripped tree)
    lab = np.array([l for l, _ in tr]); prd = np.array([p for _, p in tr])
    lim = (lab.min() * 0.7, lab.max() * 1.6)
    a.plot(lim, lim, color="#888888", lw=0.8, zorder=2)
    a.scatter(lab, prd, s=11, color=REC, edgecolor="black", linewidth=0.3,
              alpha=0.85, zorder=3)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(*lim); a.set_ylim(*lim)
    a.set_title("L1$\\rightarrow$L2 read sectors — RTX 5060", fontsize=7, pad=2.5)
    a.set_xlabel("NCU reference (sectors)", fontsize=6.8, labelpad=1.5)
    a.set_ylabel("counter-free prediction", fontsize=6.8, labelpad=1.5)
    a.annotate("accepted — fixed\ntest split: geomean\nrel. error "
               "$2.7{\\times}10^{-12}$ "
               "($n{=}39$),\nno fitted parameters",
               (0.035, 0.875), xycoords="axes fraction", ha="left", va="top",
               fontsize=6.0, color=REC, weight="bold")
    # inset: the exactness is a visible shape, not just a printed number --
    # per-cell relative error over the full trace cohort
    rel = np.abs(prd - lab) / np.maximum(lab, 1e-12)
    edges = [("exact", int((rel < 1e-9).sum())),
             ("$<10^{-4}$", int(((rel >= 1e-9) & (rel < 1e-4)).sum())),
             ("$<10^{-2}$", int(((rel >= 1e-4) & (rel < 1e-2)).sum())),
             ("$\\geq10^{-2}$", int((rel >= 1e-2).sum()))]
    ia = a.inset_axes([0.56, 0.065, 0.41, 0.295])
    yy_in = np.arange(len(edges))[::-1]
    ia.barh(yy_in, [n for _, n in edges], 0.62, color=REC,
            edgecolor="black", linewidth=0.4)
    for y_in, (_, n) in zip(yy_in, edges):
        ia.annotate(str(n), (n + 5, y_in), ha="left", va="center",
                    fontsize=4.8, color="#111111")
    ia.set_yticks(yy_in)
    ia.set_yticklabels([e for e, _ in edges], fontsize=4.6)
    ia.set_xticks([])
    ia.set_xlim(0, 178)
    ia.grid(False)
    ia.set_title(f"per-cell rel. error ($n{{=}}{len(tr)}$)", fontsize=5.0, pad=1.5)
    for sp in ia.spines.values():
        sp.set_linewidth(0.5)
    a.tick_params(labelsize=6.5)
    _panel_tag(a, "(a)")

    # ---- (b) read + write sectors, sm_86: accepted write, read outside support ---
    tp = _load("sm_86_predictor_accuracy_recovery_v2/t2_traffic_prediction_sm_86.json",
               base=_AMP)
    if tp:
        rd = _t2_pairs(tp, "read_sectors")
        wr = _t2_pairs(tp, "write_sectors")
    else:  # layout fallback
        rd, wr = [("x", 2e4, 2e4)], [("y", 64, 64)]
    wl = np.array([l for _, l, _ in wr]); wp = np.array([p for _, _, p in wr])
    rl = np.array([l for _, l, _ in rd]); rp = np.array([p for _, _, p in rd])
    lim = (min(wl.min(), rl.min()) * 0.5, max(rl.max(), wl.max(), rp.max()) * 2.2)
    b.plot(lim, lim, color="#888888", lw=0.8, zorder=2)
    b.scatter(wl, wp, s=15, color=REC, edgecolor="black", linewidth=0.35,
              zorder=3, label="write (accepted: 19/20 exact)")
    b.scatter(rl, rp, s=17, facecolor="white", edgecolor="#333333",
              linewidth=0.8, marker="^", zorder=4,
              label="read (p90 0.51 outside support)")
    for slug, txt, dxy in (
            ("l2_persisting_window_sweep",
             "persisting-window\ncarveout: $1.93$", (0.50, 0.93)),
            ("rmsnorm_fwd_bf16",
             "full-line\nfetch: $0.51$", (0.80, 0.52))):
        cell = next(((ll, pp) for s, ll, pp in rd if s == slug), None)
        if cell:
            b.annotate(txt, xy=cell, xycoords="data",
                       xytext=dxy, textcoords="axes fraction",
                       fontsize=5.8, color="#333333", ha="center", va="top",
                       arrowprops=dict(arrowstyle="->", color="#666666", lw=0.6,
                                       shrinkB=2.5))
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(*lim); b.set_ylim(*lim)
    b.set_title("read $+$ write sectors — RTX 3090 Ti", fontsize=7, pad=2.5)
    b.set_xlabel("NCU reference (sectors)", fontsize=6.8, labelpad=1.5)
    b.legend(loc="upper left", fontsize=5.5, borderpad=0.25, handletextpad=0.2,
             labelspacing=0.3, bbox_to_anchor=(0.015, 0.875))
    b.tick_params(labelsize=6.5)
    _panel_tag(b, "(b)")

    # ---- (c) L1 hit rate, sm_86: outside support, with the diagnosed cause
    l1cells = _sm86_l1_cells() or [(0.5, 0.5)]
    t = np.array([x for x, _ in l1cells]); p = np.array([y for _, y in l1cells])
    byp = t < 0.02
    c.plot([0, 1], [0, 1], color="#888888", lw=0.8, zorder=2)
    c.scatter(t[~byp], p[~byp], s=14, color="#9aa7b1", edgecolor="black",
              linewidth=0.35, zorder=3)
    c.scatter(t[byp], p[byp], s=16, facecolor="white", edgecolor="#8c1c1c",
              linewidth=0.9, zorder=4)
    c.annotate("cache-bypass kernels (12, $\\circ$): true\nrate ${\\approx}\\,0$, "
               "model treats them as\ncacheable — every nonzero prediction\n"
               "is unbounded relative error",
               xy=(0.028, 0.38), xytext=(0.13, 0.97), xycoords="data",
               textcoords="axes fraction", fontsize=5.7, color="#8c1c1c",
               ha="left", va="top",
               arrowprops=dict(arrowstyle="-", color="#8c1c1c", lw=0.6,
                               shrinkB=2))
    # stacked-marker counts, so visible glyphs reconcile with the stated n
    for (cx, cy), (tx, ty), n_st, col in (
            ((0.0, 0.0413), (0.030, 0.012), 7, "#8c1c1c"),
            ((0.0, 0.056), (0.030, 0.085), 3, "#8c1c1c"),
            ((0.5, 0.671), (0.525, 0.671), 4, "#555555")):
        c.annotate(f"$\\times${n_st}", (tx, ty), fontsize=4.8,
                   color=col, ha="left", va="center", zorder=5)
    c.set_xlim(-0.04, 1.04); c.set_ylim(-0.04, 1.04)
    c.set_xticks([0, 0.5, 1]); c.set_yticks([0, 0.5, 1])
    c.set_title("L1 hit rate — RTX 3090 Ti", fontsize=7, pad=2.5)
    c.set_xlabel("NCU reference (fraction)", fontsize=6.8, labelpad=1.5)
    c.annotate("outside support: GMRE $0.89$ ($n{=}30$)", (0.97, 0.165),
               xycoords="axes fraction", ha="right", va="bottom",
               fontsize=6.2, color="#111111", weight="bold")
    c.tick_params(labelsize=6.5)
    _panel_tag(c, "(c)")

    # ---- (d) L2 hit ratio, sm_120: the erasure, per family ----------------
    l2c = _sm120_l2_cells() or [("memory_stress", 0.97, 0.97)]
    FAM = {  # paper-facing family names + verdict colouring (fig_l2_family)
        "memory_stress": ("memory streaming (recoverable)", REC, "o", REC),
        "activation_norm": ("reduction / normalize (above threshold)",
                            C["orange"], "s", C["orange"]),
        "other": ("intermediate buffers (not identifiable)", "white", "^", NON),
        "tensor_compute": ("matrix multiply (not identifiable)", NON, "v", NON),
    }
    d.plot([0, 1], [0, 1], color="#888888", lw=0.8, zorder=2)
    handles = []
    # draw the recoverable family LAST so its two collapsed low-locality
    # cells are never occluded by the erased families' markers
    for fam in ("activation_norm", "other", "tensor_compute", "memory_stress"):
        labd, fc, mk, ec = FAM[fam]
        xs = [l for f, l, _ in l2c if f == fam]
        ys = [p for f, _, p in l2c if f == fam]
        z = 4 if fam == "memory_stress" else 3
        d.scatter(xs, ys, s=16, facecolor=fc,
                  edgecolor="black" if fc != "white" else ec,
                  linewidth=0.4 if fc != "white" else 0.9, marker=mk, zorder=z)
        handles.append(plt.Line2D([], [], marker=mk, ls="", ms=4.2,
                                  mfc=fc, mec="black" if fc != "white" else ec,
                                  mew=0.4 if fc != "white" else 0.9, label=labd))
    handles.insert(0, handles.pop())  # legend order: recoverable family first
    chip_d = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none",
                  alpha=0.88)
    d.annotate("staged-reuse families collapse\nto ${\\approx}\\,0$: reuse staged\n"
               "on-chip, invisible to the trace",
               (0.985, 0.50), xycoords="axes fraction", ha="right", va="center",
               fontsize=5.9, color=NON, bbox=chip_d)
    d.annotate("$r{=}0.38$ ($n{=}29$) — not identifiable", (0.985, 0.315),
               xycoords="axes fraction", ha="right", va="bottom",
               fontsize=6.2, color="#111111", weight="bold", bbox=chip_d)
    d.set_xlim(-0.04, 1.04); d.set_ylim(-0.04, 1.04)
    d.set_xticks([0, 0.5, 1]); d.set_yticks([0, 0.5, 1])
    d.set_title("L2 hit ratio — RTX 5060", fontsize=7, pad=2.5)
    d.set_xlabel("NCU reference (fraction)", fontsize=6.8, labelpad=1.5)
    d.legend(handles=handles, loc="upper left", fontsize=5.2, borderpad=0.3,
             handletextpad=0.2, labelspacing=0.3, bbox_to_anchor=(0.015, 0.875),
             frameon=True, framealpha=0.95, edgecolor="#dddddd")
    d.tick_params(labelsize=6.5)
    _panel_tag(d, "(d)")
    save(fig, "fig_microbench.pdf")


if __name__ == "__main__":
    _verify_against_artifacts()
    # The overview schematic is a hand-authored TikZ figure inline in main.tex;
    # the data figures below remain artifact-asserted. fig_recoverable was cut
    # for the 8-page budget; fig_map and fig_l2_family remain available for
    # regeneration. The map verdict is carried by the §map prose, and the
    # per-family L2 result is now the
    # normalized main-body table tab:l2family (its medians, x-gate ratios,
    # and cross-check are asserted in _verify_against_artifacts). The cut
    # functions are kept for regeneration.
    fig_timing()
    fig_framework()
    fig_microbench()
    print("all figures done")
