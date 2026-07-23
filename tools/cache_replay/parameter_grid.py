"""Pre-registered Blackwell-L2 parameter grid and staged stratified-CV selection.

The unknown Blackwell L2 geometry (effective capacity fraction, associativity,
replacement policy, slice count, slice-hash) is selected from a fixed,
pre-registered discrete grid of at most ``GRID_SIZE_CAP`` candidate
configurations. No free-form machine-learning model is fit over the 30 cells;
only a one-dimensional monotone residual correction (see ``residual.py``) is
applied on top of the selected mechanistic configuration.

Staged search (keeps the enumerated grid small and the selection interpretable):
  * stage one  -- effective-capacity scale x associativity x replacement,
                  with a single slice and the identity slice-hash.
  * stage two  -- fix stage-one's winner; vary slice count x slice-hash.

Selection rule (pre-registered, deterministic):
  * score each candidate by stratified cross-validation on the primary metric
    (median absolute hit-rate error), seed 20260527;
  * the simplest candidate whose CV score is within two percent of the best
    wins (the simpler-within-two-percent tie-break).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # type: ignore[import-not-found]  # noqa: E402

GRID_SIZE_CAP = 75
CV_SEED = 20260527
SIMPLER_WITHIN_FRACTION = 0.02  # simplest config within two percent of best wins

# Pre-registered candidate axes (frozen).
CAPACITY_SCALES = (0.75, 1.0, 1.25)
ASSOCIATIVITIES = (8, 16, 32)
REPLACEMENTS = ("LRU", "SRRIP", "BIP")
# Stage-two slice geometry. num_slices == 1 collapses to the single-slice model
# regardless of slice-hash, so it is represented once (stage one) and stage two
# explores genuine multi-slice partitioning only.
SLICE_COUNTS = (2, 4, 8)
SLICE_HASH_CANDIDATES = (
    "line_mod", "sector_mod", "xor_folded", "knuth_mult", "bit_reverse",
)
# Prefetch stays off this round: no microbench evidence admits next-line/
# next-sector prefetch, so admitting it would be an unsupported claim.
PREFETCH = "off"


def _build_grid(capacity_bytes: int) -> tuple[scr.CacheReplayConfig, ...]:
    """Enumerate the frozen candidate grid as a tuple of configs."""
    grid: list[scr.CacheReplayConfig] = []
    # stage one: capacity-scale x associativity x replacement, single slice.
    for cap_scale in CAPACITY_SCALES:
        for assoc in ASSOCIATIVITIES:
            for repl in REPLACEMENTS:
                grid.append(scr.CacheReplayConfig(
                    capacity_bytes=capacity_bytes,
                    associativity=assoc,
                    replacement_policy=repl,
                    num_slices=1,
                    slice_hash_id="line_mod",
                    capacity_scale=cap_scale,
                    prefetch=PREFETCH,
                ))
    # stage two: multi-slice geometry x slice-hash, anchored at the canonical
    # mid-grid policy (associativity 16, LRU, scale 1.0). The selector below
    # re-anchors stage two on stage one's actual winner; this enumeration
    # records the full reachable candidate set for the grid-size guard.
    for num_slices in SLICE_COUNTS:
        for slice_hash in SLICE_HASH_CANDIDATES:
            grid.append(scr.CacheReplayConfig(
                capacity_bytes=capacity_bytes,
                associativity=16,
                replacement_policy="LRU",
                num_slices=num_slices,
                slice_hash_id=slice_hash,
                capacity_scale=1.0,
                prefetch=PREFETCH,
            ))
    return tuple(grid)


def grid_for_device(capacity_bytes: int) -> tuple[scr.CacheReplayConfig, ...]:
    grid = _build_grid(capacity_bytes)
    if len(grid) > GRID_SIZE_CAP:
        raise ValueError(
            f"parameter grid has {len(grid)} configs, exceeds the "
            f"pre-registered cap of {GRID_SIZE_CAP}"
        )
    return grid


def stage_one_configs(grid: Sequence[scr.CacheReplayConfig]):
    return [c for c in grid if c.num_slices == 1 and c.slice_hash_id == "line_mod"]


def stage_two_configs(anchor: scr.CacheReplayConfig):
    """Stage-two candidates: the multi-slice geometry **re-anchored on the
    stage-one winner**. Each candidate inherits the winner's capacity_scale,
    associativity, and replacement_policy and varies only num_slices and
    slice_hash_id (the pre-registered staged protocol). Produces
    ``len(SLICE_COUNTS) * len(SLICE_HASH_CANDIDATES)`` candidates."""
    out: list[scr.CacheReplayConfig] = []
    for num_slices in SLICE_COUNTS:
        for slice_hash in SLICE_HASH_CANDIDATES:
            out.append(scr.CacheReplayConfig(
                capacity_bytes=anchor.capacity_bytes,
                associativity=anchor.associativity,
                replacement_policy=anchor.replacement_policy,
                num_slices=num_slices,
                slice_hash_id=slice_hash,
                capacity_scale=anchor.capacity_scale,
                prefetch=anchor.prefetch,
            ))
    return out


STAGE_TWO_CANDIDATE_COUNT = len(SLICE_COUNTS) * len(SLICE_HASH_CANDIDATES)


def config_complexity(cfg: scr.CacheReplayConfig) -> tuple:
    """Lower is simpler. Used by the simpler-within-two-percent tie-break."""
    repl_rank = {"LRU": 0, "SRRIP": 1, "BIP": 2}[cfg.replacement_policy]
    hash_rank = list(scr.SLICE_HASH_IDS).index(cfg.slice_hash_id)
    return (
        cfg.num_slices,                  # fewer slices is simpler
        cfg.associativity,               # smaller associativity is simpler
        repl_rank,                       # LRU < SRRIP < BIP
        abs(cfg.capacity_scale - 1.0),   # closer to the knowable capacity
        hash_rank,                       # identity hash is simplest
    )


@dataclass(frozen=True)
class Selection:
    selected: scr.CacheReplayConfig
    best: scr.CacheReplayConfig
    best_cv_score: float
    selected_cv_score: float
    tie_break_applied: bool
    stage_one_winner: scr.CacheReplayConfig
    stage_two_anchor: scr.CacheReplayConfig
    n_candidates_evaluated: int
    evaluated_config_signatures: tuple

    def provenance(self) -> dict:
        return {
            "grid_size_cap": GRID_SIZE_CAP,
            "cv_seed": CV_SEED,
            "simpler_within_fraction": SIMPLER_WITHIN_FRACTION,
            "primary_metric": "median_absolute_hit_rate_error",
            "n_candidates_evaluated": self.n_candidates_evaluated,
            "stage_one_winner": self.stage_one_winner.provenance(),
            "stage_two_anchor": self.stage_two_anchor.provenance(),
            "evaluated_config_signatures": [list(s) for s in self.evaluated_config_signatures],
            "best_config": self.best.provenance(),
            "best_cv_score": self.best_cv_score,
            "selected_config": self.selected.provenance(),
            "selected_cv_score": self.selected_cv_score,
            "simpler_within_2pct_tie_break_applied": self.tie_break_applied,
        }


def stratified_folds(
    family_buckets: Sequence[str], n_folds: int = 5, seed: int = CV_SEED,
) -> list[int]:
    """Deterministic stratified fold assignment by family bucket (hash-based
    round-robin per bucket, no stdlib randomness). Returns a fold index per
    cell."""
    import hashlib

    order: dict[str, list[int]] = {}
    for idx, bucket in enumerate(family_buckets):
        order.setdefault(bucket, []).append(idx)
    fold_of = [0] * len(family_buckets)
    for bucket, idxs in order.items():
        # deterministic per-bucket order via SHA-256 of (seed, bucket, idx)
        def _key(i: int) -> int:
            h = hashlib.sha256(f"{seed}:{bucket}:{i}".encode()).digest()
            return int.from_bytes(h[:8], "big")
        for rank, i in enumerate(sorted(idxs, key=_key)):
            fold_of[i] = rank % n_folds
    return fold_of


def _config_signature(cfg: scr.CacheReplayConfig) -> tuple:
    p = cfg.provenance()
    return (
        p["capacity_scale"], p["associativity"], p["replacement_policy"],
        p["num_slices"], p["slice_hash_id"],
    )


def select_config(
    grid: Sequence[scr.CacheReplayConfig],
    *,
    predict_fn: Callable[[scr.CacheReplayConfig], Sequence[float]],
    labels: Sequence[float],
    family_buckets: Sequence[str],
    cv_score_fn: Callable[[Sequence[float], Sequence[float], Sequence[str]], float],
) -> Selection:
    """Run the pre-registered STAGED stratified-CV selection.

    Stage one evaluates the 27 single-slice configs (capacity_scale x
    associativity x replacement). Stage two RE-ANCHORS on the stage-one winner:
    it inherits the winner's capacity_scale/associativity/replacement and varies
    only num_slices/slice_hash. ``predict_fn(config) -> per-cell predictions``
    supplies predictions for any candidate (the re-anchored stage-two configs
    need not pre-exist in any matrix). ``cv_score_fn`` returns the
    cross-validated primary-metric score (lower is better).
    """
    key = _config_signature

    def score(cfg: scr.CacheReplayConfig) -> float:
        return cv_score_fn(predict_fn(cfg), labels, family_buckets)

    s1 = stage_one_configs(grid)
    s1_scored = [(c, score(c)) for c in s1]
    s1_best = min(s1_scored, key=lambda cs: (cs[1], config_complexity(cs[0])))[0]

    # stage two re-anchored on the actual stage-one winner
    s2 = stage_two_configs(s1_best)

    dedup: dict[tuple, tuple] = {}
    for c in (s1 + s2):
        if key(c) not in dedup:
            dedup[key(c)] = (c, score(c))
    candidates = list(dedup.values())

    if len(candidates) > GRID_SIZE_CAP:
        raise ValueError(
            f"staged search evaluated {len(candidates)} configs, exceeds the "
            f"pre-registered cap of {GRID_SIZE_CAP}"
        )

    best_cfg, best_score = min(candidates, key=lambda cs: (cs[1], config_complexity(cs[0])))
    threshold = best_score * (1.0 + SIMPLER_WITHIN_FRACTION) + 1e-12
    within = [(c, sc) for c, sc in candidates if sc <= threshold]
    selected_cfg = min(within, key=lambda cs: config_complexity(cs[0]))[0]
    tie_break = key(selected_cfg) != key(best_cfg)

    return Selection(
        selected=selected_cfg,
        best=best_cfg,
        best_cv_score=best_score,
        selected_cv_score=dict((key(c), sc) for c, sc in candidates)[key(selected_cfg)],
        tie_break_applied=tie_break,
        stage_one_winner=s1_best,
        stage_two_anchor=s1_best,
        n_candidates_evaluated=len(candidates),
        evaluated_config_signatures=tuple(dedup.keys()),
    )
