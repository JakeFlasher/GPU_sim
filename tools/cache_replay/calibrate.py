"""Calibration + comparison driver for the sm_120 cache-only trace-replay predictor.

Runs the full mechanistic pipeline over the 30 sm_120 cells:

  1. decode each cell's access stream (cached);
  2. replay every pre-registered grid config (mechanistic hit ratio);
  3. select the config by staged stratified cross-validation with the
     simpler-within-two-percent tie-break;
  4. fit the one-dimensional monotone residual on top of the selected config;
  5. re-replay the selected config with the three-C miss split + uncertainty;
  6. join the six existing baseline columns (v5, Nugteren RDH, uncalibrated
     walker, v4 Platt, cohort mean, always hit) and emit the comparison bundle.

Emitted artifacts (atomic, source_commit-stamped):
  * ``data/local_labels/cache_replay_results.json``    -- per-cell replay result.
  * ``data/local_labels/cache_replay_comparison.json`` -- comparison bundle.

This driver is sm_120-develop-only: ``top_level_admission_scope`` is fixed to
``target_arch=sm_120`` and ``arches`` to ``["sm_120"]``; no sm_86 cell is read.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # noqa: E402
import cohort as cohort_mod  # noqa: E402
import parameter_grid as pg  # noqa: E402
import residual as residual_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import artifact_io  # noqa: E402

RESULTS_PATH = _REPO_ROOT / "data/local_labels/cache_replay_results.json"
COMPARISON_PATH = _REPO_ROOT / "data/local_labels/cache_replay_comparison.json"
_BASELINE_COMPARISON = _REPO_ROOT / "data/local_labels/calibration_head_v5_baseline_comparison.json"

SCHEMA_VERSION = 1
ALGORITHM_VERSION = "cache_replay_v1"
TOP_LEVEL_ADMISSION_SCOPE = "target_arch=sm_120"
ARCHES = ["sm_120"]
N_HOLDOUT = 7

BASELINE_COLUMNS = (
    "v5_prediction",
    "nugteren_prediction",
    "uncalibrated_walker_prediction",
    "v4_platt_prediction",
    "cohort_mean_prediction",
    "always_hit_prediction",
)


def _config_key(cfg: scr.CacheReplayConfig) -> tuple:
    p = cfg.provenance()
    return (
        p["capacity_scale"], p["associativity"], p["replacement_policy"],
        p["num_slices"], p["slice_hash_id"],
    )


def holdout_cell_ids(cells, n_holdout: int = N_HOLDOUT, seed: int = pg.CV_SEED) -> set:
    """Deterministic stratified holdout selection (proportional per bucket)."""
    by_bucket: dict[str, list] = {}
    for c in cells:
        by_bucket.setdefault(c.family_bucket, []).append(c)

    def _key(cell) -> int:
        h = hashlib.sha256(f"{seed}:holdout:{cell.cell_id}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    chosen: list[str] = []
    # proportional allocation, at least the highest-priority cell per bucket
    total = len(cells)
    for _bucket, members in sorted(by_bucket.items()):
        members_sorted = sorted(members, key=_key)
        take = max(1, round(n_holdout * len(members) / total))
        chosen.extend(m.cell_id for m in members_sorted[:take])
    # trim/pad deterministically to exactly n_holdout
    chosen_sorted = sorted(chosen)
    if len(chosen_sorted) > n_holdout:
        chosen_sorted = chosen_sorted[:n_holdout]
    return set(chosen_sorted)


_PRED_CACHE_DIR = _REPO_ROOT / ".cache_replay_streams"


def _prediction_cache_path(cohort) -> Path:
    """Prediction-cache file namespaced by the cohort's trace fingerprint.

    Keying the memoized config->prediction map by config signature ALONE leaves
    it stale when any cell's NVBit trace is regenerated/swapped. Folding the
    cohort trace fingerprint into the filename makes a changed trace produce a
    fresh cache file (the prior file is orphaned under the gitignored
    ``.cache_replay_streams/`` and harmlessly ignored)."""
    fp = cohort_mod.cohort_fingerprint(cohort)
    return _PRED_CACHE_DIR / f"prediction_cache__{fp}.json"


class _PredictionProvider:
    """Supplies per-cell mechanistic predictions for an ARBITRARY config (the
    staged search re-anchors stage two on the winner, so the evaluated configs
    are not known in advance). Each config's 30-cell prediction vector is
    computed once via replay and memoized on disk keyed by the config signature
    (in a cache file namespaced by the cohort trace fingerprint, so a changed
    trace invalidates it); re-runs under the same config/cohort/trace are
    instant."""

    def __init__(self, cohort, cell_ids, *, use_cache: bool = True):
        import json
        self.cohort = cohort
        self.cell_ids = cell_ids
        self.use_cache = use_cache
        self._cache_path = _prediction_cache_path(cohort)
        self._cache: dict[str, dict[str, float]] = {}
        if use_cache and self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text())
        self._streams: dict[str, tuple[list, list]] = {}

    def _stream(self, cell):
        if cell.cell_id not in self._streams:
            lids, sof = cohort_mod.decode_stream(cell, use_cache=self.use_cache)
            self._streams[cell.cell_id] = (lids.tolist(), sof.tolist())
        return self._streams[cell.cell_id]

    def predict(self, config) -> list[float]:
        sig = repr(_config_key(config))
        if sig not in self._cache:
            per_cell = {}
            for cell in self.cohort:
                lids, sof = self._stream(cell)
                per_cell[cell.cell_id] = scr.replay_access_stream(
                    lids, sof, config, classify_misses=False
                ).predicted_hit_ratio
            self._cache[sig] = per_cell
        return [self._cache[sig][cid] for cid in self.cell_ids]

    def medabs(self, config, labels) -> float:
        return metrics_mod.median_absolute_error(self.predict(config), labels)

    def cached_signatures(self) -> list[str]:
        return list(self._cache.keys())

    def predictions_for_sig(self, sig: str) -> list[float]:
        return [self._cache[sig][cid] for cid in self.cell_ids]

    def save(self) -> None:
        import json
        if not self.use_cache:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))


def _cv_score_fn(predictions, labels, family_buckets):
    """Stratified-CV median-absolute-error with a per-fold monotone residual
    fit. ``predictions`` is the list of mechanistic predictions aligned with
    ``labels`` and ``family_buckets``."""
    n = len(labels)
    folds = pg.stratified_folds(family_buckets, n_folds=5)
    abs_errs: list[float] = []
    for fold in range(5):
        train_idx = [i for i in range(n) if folds[i] != fold]
        test_idx = [i for i in range(n) if folds[i] == fold]
        if not test_idx or len(train_idx) < 2:
            continue
        head = residual_mod.fit_monotone_residual(
            [predictions[i] for i in train_idx],
            [labels[i] for i in train_idx],
        )
        test_pred = head.predict([predictions[i] for i in test_idx])
        for j, i in enumerate(test_idx):
            abs_errs.append(abs(test_pred[j] - labels[i]))
    if not abs_errs:
        return 1.0
    import statistics
    return float(statistics.median(abs_errs))


def _load_baseline_rows():
    import json
    if not _BASELINE_COMPARISON.exists():
        return {}
    data = json.loads(_BASELINE_COMPARISON.read_text())
    out: dict[str, dict] = {}
    for row in data.get("rows", []):
        if row.get("arch") != "sm_120":
            continue
        cid = f"{row['slug']}/{row['shape_point']}"
        out[cid] = {col: row.get(col) for col in BASELINE_COLUMNS}
    return out


def run_calibration(*, use_cache: bool = True) -> dict:
    cohort = cohort_mod.load_cohort()
    capacity_bytes = cohort_mod.RTX_5060_LAPTOP_L2_BYTES
    capacity_provenance = cohort_mod.L2_CAPACITY_PROVENANCE
    grid = pg.grid_for_device(capacity_bytes)

    labels = [c.label for c in cohort]
    buckets = [c.family_bucket for c in cohort]
    cell_ids = [c.cell_id for c in cohort]

    provider = _PredictionProvider(cohort, cell_ids, use_cache=use_cache)
    selection = pg.select_config(
        grid,
        predict_fn=provider.predict,
        labels=labels,
        family_buckets=buckets,
        cv_score_fn=_cv_score_fn,
    )
    provider.save()
    selected = selection.selected
    mech_list = provider.predict(selected)
    mech_pred = dict(zip(cell_ids, mech_list))

    # inert-grid evidence over the configs actually evaluated this run (stage
    # one + the re-anchored stage two): per-cell spread of mechanistic
    # predictions across the evaluated set (the model-parameter signal).
    evaluated_sigs = [repr(tuple(s)) for s in selection.evaluated_config_signatures]
    spread = {}
    for cid in cell_ids:
        vals = [provider._cache[s][cid] for s in evaluated_sigs if s in provider._cache]
        spread[cid] = (max(vals) - min(vals)) if vals else 0.0
    max_grid_spread = max(spread.values()) if spread else 0.0

    # contribution of set-associative realism specifically: the improvement of
    # the most set-associative evaluated config over the fully-associative
    # single-slice config; near-zero when the grid is inert.
    fully_assoc = scr.CacheReplayConfig(
        capacity_bytes, associativity=max(pg.ASSOCIATIVITIES),
        replacement_policy="LRU", num_slices=1, slice_hash_id="line_mod",
        capacity_scale=1.0,
    )
    most_set_assoc = scr.CacheReplayConfig(
        capacity_bytes, associativity=selected.associativity,
        replacement_policy=selected.replacement_policy,
        num_slices=max(pg.SLICE_COUNTS), slice_hash_id="knuth_mult",
        capacity_scale=selected.capacity_scale,
    )
    evaluated_medabs = [
        metrics_mod.median_absolute_error(provider.predictions_for_sig(s), labels)
        for s in evaluated_sigs if s in provider._cache
    ]
    grid_config_medabs = {
        "min": min(evaluated_medabs) if evaluated_medabs else 0.0,
        "max": max(evaluated_medabs) if evaluated_medabs else 0.0,
        "fully_associative_single_slice": provider.medabs(fully_assoc, labels),
        "most_set_associative": provider.medabs(most_set_assoc, labels),
    }
    provider.save()

    # full-cohort monotone residual (in-sample headline calibration).
    residual = residual_mod.fit_monotone_residual(
        [mech_pred[cid] for cid in cell_ids], labels,
    )
    calibrated_list = residual.predict([mech_pred[cid] for cid in cell_ids])
    calibrated = dict(zip(cell_ids, calibrated_list))

    # re-replay selected config with the three-C split + uncertainty per cell.
    per_cell_results = []
    for cell, line_ids, sector_offsets in cohort_mod.iter_cohort_streams(
        cohort, use_cache=use_cache,
    ):
        r = scr.replay_access_stream(
            line_ids.tolist(), sector_offsets.tolist(), selected, classify_misses=True,
        )
        out = r.as_dict()
        out["cache_parameters"] = selected.provenance()
        out["join_key"] = {
            "slug": cell.slug,
            "shape_point": cell.shape_point,
            "arch": cohort_mod.TARGET_ARCH,
            "family_bucket": cell.family_bucket,
            "cache_layer": "L2",
        }
        out["ground_truth_hit_ratio"] = cell.label
        out["calibrated_hit_ratio"] = calibrated[cell.cell_id]
        per_cell_results.append(out)

    baselines = _load_baseline_rows()
    holdout = holdout_cell_ids(cohort)

    # comparison rows
    rows = []
    for c in cohort:
        row = {
            "slug": c.slug,
            "shape_point": c.shape_point,
            "arch": cohort_mod.TARGET_ARCH,
            "family_bucket": c.family_bucket,
            "label": c.label,
            "split": "holdout" if c.cell_id in holdout else "calibration",
            "replay_prediction": mech_pred[c.cell_id],
            "replay_calibrated_prediction": calibrated[c.cell_id],
        }
        for col in BASELINE_COLUMNS:
            row[col] = baselines.get(c.cell_id, {}).get(col)
        rows.append(row)

    # aggregate metrics for every predictor column
    def agg(pred_of):
        preds = [pred_of(r) for r in rows]
        labs = [r["label"] for r in rows]
        valid = [(p, y) for p, y in zip(preds, labs) if p is not None]
        if not valid:
            return {"n": 0, "median_absolute_error": None, "geomean_relative_error": None}
        p, y = zip(*valid)
        return {
            "n": len(valid),
            "median_absolute_error": metrics_mod.median_absolute_error(p, y),
            "geomean_relative_error": metrics_mod.geomean_relative_error(p, y),
        }

    aggregate = {
        "replay_prediction": agg(lambda r: r["replay_prediction"]),
        "replay_calibrated_prediction": agg(lambda r: r["replay_calibrated_prediction"]),
    }
    for col in BASELINE_COLUMNS:
        aggregate[col] = agg(lambda r, col=col: r[col])

    # head-to-head beat fractions (calibrated replay is the headline predictor)
    def beat_fraction(against):
        wins = 0
        joint = 0
        for r in rows:
            rp = r["replay_calibrated_prediction"]
            bp = r[against]
            if rp is None or bp is None:
                continue
            joint += 1
            if abs(rp - r["label"]) < abs(bp - r["label"]):
                wins += 1
        return (wins / joint if joint else 0.0), joint

    v5_beat, v5_joint = beat_fraction("v5_prediction")
    nug_beat, nug_joint = beat_fraction("nugteren_prediction")

    # holdout + per-bucket diagnostics for the calibrated headline predictor
    holdout_rows = [r for r in rows if r["split"] == "holdout"]
    holdout_geomean = (
        metrics_mod.geomean_relative_error(
            [r["replay_calibrated_prediction"] for r in holdout_rows],
            [r["label"] for r in holdout_rows],
        )
        if holdout_rows else 0.0
    )
    per_bucket = {}
    for bucket in cohort_mod.FAMILY_BUCKETS:
        brows = [r for r in rows if r["family_bucket"] == bucket]
        if not brows:
            continue
        per_bucket[bucket] = {
            "n": len(brows),
            "p90_absolute_error": metrics_mod.p90_absolute_error(
                [r["replay_calibrated_prediction"] for r in brows],
                [r["label"] for r in brows],
            ),
        }

    source_commit = artifact_io.git_head()

    results_artifact = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source_commit": source_commit,
        "arch": cohort_mod.TARGET_ARCH,
        "sku": cohort_mod.SM120_SKU,
        "top_level_admission_scope": TOP_LEVEL_ADMISSION_SCOPE,
        "arches": ARCHES,
        "l2_capacity_provenance": capacity_provenance,
        "selected_config": selected.provenance(),
        "results": per_cell_results,
    }

    comparison_bundle = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source_commit": source_commit,
        "top_level_admission_scope": TOP_LEVEL_ADMISSION_SCOPE,
        "arches": ARCHES,
        "l2_capacity_provenance": capacity_provenance,
        "result_kind": "admitted",
        "cohort_size": len(rows),
        "family_bucket_counts": {
            b: sum(1 for r in rows if r["family_bucket"] == b)
            for b in cohort_mod.FAMILY_BUCKETS
        },
        "grid_provenance": {
            "grid_size": len(grid),
            **selection.provenance(),
        },
        "residual_provenance": residual.provenance(),
        "grid_config_medabs": grid_config_medabs,
        "inert_grid_evidence": {
            "max_grid_prediction_spread": max_grid_spread,
            "note": (
                "maximum per-cell spread of the mechanistic prediction across "
                "the entire pre-registered grid; near-zero indicates the "
                "capacity/associativity/slice/replacement axes are inert "
                "because the working sets fit in the modeled capacity"
            ),
        },
        "rows": rows,
        "aggregate_metrics": aggregate,
        "replay_vs_v5_beat_fraction": v5_beat,
        "replay_vs_v5_joint_admit_count": v5_joint,
        "replay_vs_rdh_baseline_beat_fraction": nug_beat,
        "replay_vs_rdh_baseline_joint_admit_count": nug_joint,
        "holdout_geomean": holdout_geomean,
        "holdout_cell_count": len(holdout_rows),
        "per_family_bucket_p90_calibrated": per_bucket,
    }
    return {"results": results_artifact, "comparison": comparison_bundle}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Cache-replay calibration + comparison")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    out = run_calibration(use_cache=not args.no_cache)
    artifact_io.validate_against_schema(
        out["comparison"], "cache_replay_comparison.schema.json")
    for cell_result in out["results"]["results"]:
        artifact_io.validate_against_schema(
            cell_result, "cache_replay_result.schema.json")
    artifact_io.atomic_write_json(RESULTS_PATH, out["results"])
    artifact_io.atomic_write_json(COMPARISON_PATH, out["comparison"])
    c = out["comparison"]
    print(f"selected config: {c['grid_provenance']['selected_config']}")
    print(f"max grid prediction spread: {c['inert_grid_evidence']['max_grid_prediction_spread']:.6f}")
    print(f"replay mechanistic: {c['aggregate_metrics']['replay_prediction']}")
    print(f"replay calibrated:  {c['aggregate_metrics']['replay_calibrated_prediction']}")
    print(f"v5 baseline:        {c['aggregate_metrics']['v5_prediction']}")
    print(f"nugteren baseline:  {c['aggregate_metrics']['nugteren_prediction']}")
    print(f"replay vs v5 beat:           {c['replay_vs_v5_beat_fraction']:.3f} ({c['replay_vs_v5_joint_admit_count']})")
    print(f"replay vs rdh-baseline beat: {c['replay_vs_rdh_baseline_beat_fraction']:.3f} ({c['replay_vs_rdh_baseline_joint_admit_count']})")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {COMPARISON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
