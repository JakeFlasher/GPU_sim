#!/usr/bin/env python3
"""GCoM-lite analytical cycles evaluator: an interval/CPI-stack model
over the expanded timing cohort, composed from COMMITTED surfaces —
NO per-cycle simulation machinery exists here (analytical closed
forms only; the per-instruction loop below iterates STATIC SASS
records to average control bits, never simulated time).

Per admitted cohort cell, four INDIVIDUALLY INSPECTABLE component
terms (each row records its inputs and formula):

- ``compute``: per-SM warp instructions / the per-SM issue width
  (4 schedulers x 1 warp-instruction per cycle — architectural).
- ``stall``: per-SM warp instructions x the kernel's MEAN STATIC
  stall cycles per instruction (the SASS control-bit readout,
  extracted from the cell's PINNED binary through the toolkit-gated
  extractor), overlapped across co-resident warps.
- ``memory``: the warp-pool walker's OWN miss stream (miss_count +
  bypass_count — the same composition the traffic lane validated
  against the live xbar sector labels) x the hit-weighted downstream
  latency (the card's L2/DRAM latencies weighted by the frozen v5
  head's committed L2 ``raw_prediction`` feature — label fields are
  NEVER read), divided across SMs and overlapped by the
  card-resolved MSHR depth (the batching correction).
- ``imbalance``: the wave-quantization remainder; 0.0 with an
  explicit note under the round-robin uniform-distribution
  assumption the trace replay uses.

The analytical prior is ``max(compute + stall, memory) +
imbalance`` (issue/stall serialize per warp and overlap with the
memory stream). A ridge residual (log space, lambda frozen) fits on
the FIT POOL ONLY: cohort cells that are NOT sealed-holdout members
(split-aware; holdout-member timing cells receive predictions but
their labels are NEVER consumed here — they are the next lane's
single-touch scoring surface).

Deploy-legality (the standing demotion discipline extends to this
model): the inference feature vector passes the registry's
label-leakage guard AND the forbidden-timing-field guard imported
from the existing timing lane; the artifact declares ONE deployable
channel with no same-slug substitution path anywhere in the module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
for p in (str(REPO_ROOT), str(_HERE),
          str(REPO_ROOT / "tools" / "sass_control"),
          str(REPO_ROOT / "tools" / "local_labels"),
          str(REPO_ROOT / "tools" / "cache_predictor")):
    if p not in sys.path:
        sys.path.insert(0, p)

from extractor import extract_for_binary  # noqa: E402
from l1_warp_pool_walker import (  # noqa: E402
    resolve_latencies,
    resolve_mshr_count,
)
from timing_cycle_predictor import (  # noqa: E402
    FORBIDDEN_FEATURE_FIELDS,
)

DEFAULT_COHORT = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort"
    / "sm86_timing_cohort_expanded.json"
)
DEFAULT_SPLIT = (
    REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_split.json"
)
DEFAULT_EMISSIONS = (
    REPO_ROOT / "data" / "local_labels"
    / "replay_count_emissions_sm_86.json"
)
DEFAULT_WALKER = (
    REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "warp_pool_l1_per_admitted_cell.json"
)
DEFAULT_V5_HEAD = (
    REPO_ROOT / "data" / "local_labels" / "sm_86_v5_head.json"
)
DEFAULT_CARD = (
    REPO_ROOT / "data" / "pchase" / "sm_86_static_card.json"
)
DEFAULT_WITNESS = (
    REPO_ROOT / "data" / "sass_control"
    / "pc_join_witness_sm_86.json"
)
DEFAULT_OUT = (
    REPO_ROOT / "data" / "local_labels" / "timing_predictions"
    / "gcom_lite_cycles_sm_86.json"
)
DEFAULT_SCHEMA = _HERE / "gcom_lite_cycles.schema.json"

_T1_INST = "smsp__inst_executed.sum"
_CYCLES_LABEL = "gpc__cycles_elapsed.max"

MODEL_VERSION = "gcom_lite_interval_cpi_stack_v1"
DEPLOYABLE_CHANNEL = (
    "deployable_gcom_lite_interval_no_same_slug_substitution"
)
ISSUE_WIDTH_PER_SM = 4.0   # 4 schedulers x 1 warp-inst/cycle (arch)
RIDGE_LAMBDA = 10.0

FEATURE_NAMES = (
    "bias",
    "log1p_inst_executed",
    "log1p_walker_miss_stream_sectors",
    "predicted_l1_hit_rate",
    "predicted_l2_hit_rate_raw",
    "mean_static_stall_cycles",
    "co_resident_warps_over_48",
    "active_sm_over_84",
)


class EvaluatorInputError(Exception):
    """A committed input is missing or fails its value audit —
    the cell lands typed, never silently modeled."""


class DeployLegalityError(Exception):
    """A label-only or forbidden field reached the inference
    feature vector."""


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def assert_features_deploy_legal(
    feature_names: tuple[str, ...],
) -> None:
    """Both standing guards: the registry label-leakage check and
    the timing lane's forbidden-field set."""
    forbidden = set(feature_names) & set(FORBIDDEN_FEATURE_FIELDS)
    if forbidden:
        raise DeployLegalityError(
            f"forbidden timing fields in the inference feature "
            f"vector: {sorted(forbidden)}"
        )
    from metric_registry import load_registry

    load_registry().assert_no_label_leakage(set(feature_names))


def _dram_latency_cycles(card_path: Path) -> float:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    prop = card.get("properties", {}).get("dram_latency_cycles")
    if not prop or prop.get("kind") != "value":
        raise EvaluatorInputError(
            "card dram_latency_cycles unavailable; an unsourced "
            "latency constant is rejected")
    raw = prop.get("value")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) \
            or not math.isfinite(raw) or raw <= 0:
        raise EvaluatorInputError(
            f"dram_latency_cycles value {raw!r} fails the value "
            "audit")
    return float(raw)


def _mean_static_stall(
    binary: Path, *, kernel_names: list[str] | None,
) -> tuple[float, dict]:
    """Mean control-bit stall cycles per decoded instruction of the
    cell's kernel function(s); falls back to all functions with a
    recorded note when the witness gives no kernel names."""
    extracted = extract_for_binary(binary, arch="sm_86")
    functions = extracted["functions"]
    selected: dict[str, list] = {}
    if kernel_names:
        for name, records in functions.items():
            if any(k in name for k in kernel_names):
                selected[name] = records
    scope_note = "witness_kernel_functions"
    if not selected:
        selected = dict(functions)
        scope_note = "all_binary_functions_fallback"
    stalls: list[int] = []
    n_unknown = 0
    for records in selected.values():
        for rec in records:
            if rec.get("status") == "decoded":
                stalls.append(int(rec["control"]["stall"]))
            else:
                n_unknown += 1
    if not stalls:
        raise EvaluatorInputError(
            f"no decoded instructions for {binary.name} in scope "
            f"{scope_note}")
    return (
        sum(stalls) / len(stalls),
        {
            "scope": scope_note,
            "n_functions": len(selected),
            "n_decoded": len(stalls),
            "n_unknown_encoding": n_unknown,
            "toolkit_version": extracted.get("toolkit_version"),
        },
    )


def _ridge_fit(
    rows: list[tuple[list[float], float]], lam: float,
) -> list[float]:
    """Ridge solve (X^T X + lam I) b = X^T y via Gaussian
    elimination (no external deps; the bias column is row 0)."""
    n_features = len(rows[0][0])
    ata = [[0.0] * n_features for _ in range(n_features)]
    aty = [0.0] * n_features
    for x, y in rows:
        for i in range(n_features):
            aty[i] += x[i] * y
            for j in range(n_features):
                ata[i][j] += x[i] * x[j]
    for i in range(n_features):
        ata[i][i] += lam
    # Gaussian elimination with partial pivoting
    aug = [ata[i] + [aty[i]] for i in range(n_features)]
    for col in range(n_features):
        pivot = max(range(col, n_features),
                    key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        if abs(div) < 1e-12:
            continue
        aug[col] = [v / div for v in aug[col]]
        for r in range(n_features):
            if r != col and abs(aug[r][col]) > 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b
                          for a, b in zip(aug[r], aug[col])]
    return [aug[i][n_features] for i in range(n_features)]


def build_evaluation(
    *,
    cohort_path: Path,
    split_path: Path,
    emissions_path: Path,
    walker_path: Path,
    v5_head_path: Path,
    card_path: Path,
    witness_path: Path,
) -> dict:
    assert_features_deploy_legal(FEATURE_NAMES)

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    emissions = json.loads(
        emissions_path.read_text(encoding="utf-8"))
    walker = json.loads(walker_path.read_text(encoding="utf-8"))
    v5 = json.loads(v5_head_path.read_text(encoding="utf-8"))
    witness = json.loads(witness_path.read_text(encoding="utf-8"))

    latencies = resolve_latencies(card_path)
    mshr = resolve_mshr_count(card_path)
    dram_latency = _dram_latency_cycles(card_path)
    l2_latency = latencies.miss_latency  # the card's L2 hit latency

    holdout_keys = {
        (c["slug"], c["shape_point"],
         c["identity_tuple"]["binary_sha256"])
        for c in split["splits"]["holdout"]["cells"]
    }
    em_by = {
        (c["identity"]["slug"], c["identity"]["shape_point"],
         c["identity"]["binary_sha256"]): c
        for c in emissions["cells"]
    }
    wp_by = {
        (c["slug"], c["shape_point"], c["binary_sha256"]): c
        for c in walker["cells"]
    }
    v5_by = {
        (r["slug"], r["shape_point"]): r["features"]
        for r in v5["cohort_scope"]["admitted"]
    }
    witness_kernels: dict[tuple, list[str]] = {}
    for cell in witness.get("cells", []):
        ident = cell.get("identity", {})
        names = [
            f.get("kernel_name") for f in cell.get("functions", [])
            if f.get("kernel_name")
        ]
        witness_kernels[(ident.get("slug"),
                         ident.get("shape_point"))] = names

    def t1(em_cell, metric):
        if em_cell is None or em_cell.get(
                "cell_status", {}).get("status") != "emitted":
            return None
        m = (em_cell.get("metrics") or {}).get(metric)
        if not m or m.get("status") != "scored":
            return None
        return float(m["value"])

    rows: list[dict] = []
    fit_rows: list[tuple[list[float], float]] = []
    stall_cache: dict[str, tuple[float, dict]] = {}
    for cell in cohort["admitted"]:
        key = (cell["slug"], cell["shape_point"],
               cell["binary_sha256"])
        em_cell = em_by.get(key)
        wp_cell = wp_by.get(key)
        v5_features = v5_by.get((cell["slug"], cell["shape_point"]))
        missing: list[str] = []
        inst = t1(em_cell, _T1_INST)
        if inst is None:
            missing.append("t1_inst_executed_unscored")
        if (wp_cell is None
                or wp_cell.get("result_kind") != "scalar"):
            missing.append("walker_row_not_scalar")
        if v5_features is None or "raw_prediction" \
                not in v5_features:
            missing.append("v5_l2_raw_prediction_missing")
        binary = REPO_ROOT / cell["binary_pin_verification"][
            "binary_path"]
        if not binary.is_file():
            missing.append("pinned_binary_absent")
        if missing:
            rows.append({
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "binary_sha256": cell["binary_sha256"],
                "kernel_signature_sha256":
                    cell["kernel_signature_sha256"],
                "status": "typed",
                "typed_reason": "evaluator_input_unavailable",
                "evidence": {"missing_inputs": sorted(missing)},
            })
            continue

        cache_key = cell["binary_pin_verification"]["binary_path"]
        if cache_key not in stall_cache:
            stall_cache[cache_key] = _mean_static_stall(
                binary,
                kernel_names=witness_kernels.get(
                    (cell["slug"], cell["shape_point"])),
            )
        mean_stall, stall_evidence = stall_cache[cache_key]

        l1_hit = float(wp_cell["predicted_hit_rate"])
        l2_hit_raw = float(v5_features["raw_prediction"])
        evidence = wp_cell["evidence"]
        counters = evidence["counters"]
        co_resident = float(evidence["co_resident_warps"])
        active_sm = float(evidence["active_sm_count"])

        per_sm_inst = inst / max(active_sm, 1.0)
        compute = per_sm_inst / ISSUE_WIDTH_PER_SM
        stall = (per_sm_inst * mean_stall) / max(co_resident, 1.0)
        # The walker's own L2-bound sector stream: modeled misses
        # plus bypasses (both fetch from beyond L1 — the same
        # composition the traffic lane validated against the live
        # xbar read-sector labels).
        miss_stream_sectors = float(
            counters["miss_count"] + counters["bypass_count"])
        avg_miss_latency = (
            l2_hit_raw * l2_latency
            + (1.0 - l2_hit_raw) * dram_latency
        )
        per_sm_miss_sectors = (
            miss_stream_sectors / max(active_sm, 1.0))
        memory = (
            per_sm_miss_sectors * avg_miss_latency
            / float(mshr.count or 1)
        )
        imbalance = 0.0
        prior = max(compute + stall, memory) + imbalance

        features = [
            1.0,
            math.log1p(inst),
            math.log1p(miss_stream_sectors),
            l1_hit,
            l2_hit_raw,
            mean_stall,
            co_resident / 48.0,
            active_sm / 84.0,
        ]
        is_holdout = key in {
            (s, sp, b) for (s, sp, b) in holdout_keys}
        label_value = (
            None if is_holdout
            else cell["timing_labels"][_CYCLES_LABEL]["value"]
        )
        rows.append({
            "slug": cell["slug"],
            "shape_point": cell["shape_point"],
            "binary_sha256": cell["binary_sha256"],
            "kernel_signature_sha256":
                cell["kernel_signature_sha256"],
            "status": (
                "modeled_holdout_reserved" if is_holdout
                else "modeled_fit_pool"
            ),
            "component_terms": {
                "compute": {
                    "cycles": compute,
                    "formula": "per_sm_inst / issue_width",
                    "inputs": {
                        "inst_executed": inst,
                        "active_sm_count": active_sm,
                        "issue_width_per_sm": ISSUE_WIDTH_PER_SM,
                    },
                },
                "stall": {
                    "cycles": stall,
                    "formula": (
                        "per_sm_inst * mean_static_stall / "
                        "co_resident_warps"
                    ),
                    "inputs": {
                        "mean_static_stall_cycles": mean_stall,
                        "co_resident_warps": co_resident,
                        "static_stall_evidence": stall_evidence,
                    },
                },
                "memory": {
                    "cycles": memory,
                    "formula": (
                        "(walker_miss_count + walker_bypass_count) "
                        "/ active_sm * (l2_hit*l2_latency + "
                        "(1-l2_hit)*dram_latency) / mshr_count"
                    ),
                    "inputs": {
                        "walker_miss_count":
                            counters["miss_count"],
                        "walker_bypass_count":
                            counters["bypass_count"],
                        "predicted_l2_hit_rate_raw": l2_hit_raw,
                        "l2_latency_cycles": l2_latency,
                        "dram_latency_cycles": dram_latency,
                        "mshr_count": mshr.count,
                        "mshr_source": mshr.source,
                    },
                },
                "imbalance": {
                    "cycles": imbalance,
                    "formula": "wave_quantization_remainder",
                    "inputs": {
                        "note": (
                            "round_robin_uniform_distribution_"
                            "assumed_by_the_trace_replay"
                        ),
                    },
                },
            },
            "analytical_prior_cycles": prior,
            "feature_vector": dict(zip(FEATURE_NAMES, features)),
            # Set from ACTUAL fit-row membership below (a fit-pool
            # cell with a degenerate prior/label is excluded from
            # the fit and must not over-report participation).
            "label_consumed_for_fit": False,
            "prediction_channel": DEPLOYABLE_CHANNEL,
        })
        if not is_holdout and label_value is not None \
                and label_value > 0 and prior > 0:
            fit_rows.append(
                (features, math.log(label_value / prior)))
            rows[-1]["label_consumed_for_fit"] = True

    if not fit_rows:
        raise EvaluatorInputError(
            "no fit-eligible cohort cells; the residual cannot fit")
    coefficients = _ridge_fit(fit_rows, RIDGE_LAMBDA)

    for row in rows:
        if row["status"] == "typed":
            continue
        x = [row["feature_vector"][n] for n in FEATURE_NAMES]
        residual_log = sum(
            c * v for c, v in zip(coefficients, x))
        row["ridge_residual_log"] = residual_log
        row["predicted_cycles"] = (
            row["analytical_prior_cycles"]
            * math.exp(residual_log)
        )

    n_modeled = sum(
        1 for r in rows if r["status"] != "typed")
    return {
        "schema_version": 1,
        "artifact_kind": "gcom_lite_cycles",
        "arch": "sm_86",
        "generated_at_commit": _git_head(),
        "model_version": MODEL_VERSION,
        "deployable_channel": DEPLOYABLE_CHANNEL,
        "feature_names": list(FEATURE_NAMES),
        "issue_width_per_sm": ISSUE_WIDTH_PER_SM,
        "ridge_lambda": RIDGE_LAMBDA,
        "inputs": {
            "cohort": str(cohort_path.relative_to(REPO_ROOT)),
            "split": str(split_path.relative_to(REPO_ROOT)),
            "replay_emissions": str(
                emissions_path.relative_to(REPO_ROOT)),
            "walker_artifact": str(
                walker_path.relative_to(REPO_ROOT)),
            "v5_head": str(v5_head_path.relative_to(REPO_ROOT)),
            "card": str(card_path.relative_to(REPO_ROOT)),
            "pc_join_witness": str(
                witness_path.relative_to(REPO_ROOT)),
        },
        "fit": {
            "pool": "expanded_cohort_minus_sealed_holdout_members",
            "n_fit": len(fit_rows),
            "ridge_coefficients": dict(
                zip(FEATURE_NAMES, coefficients)),
            "target": "log(true_cycles / analytical_prior)",
            "holdout_labels_consumed": False,
        },
        "n_rows": len(rows),
        "n_modeled": n_modeled,
        "n_typed": len(rows) - n_modeled,
        "cells": sorted(
            rows, key=lambda r: (r["slug"], r["shape_point"])),
    }


def validate_artifact_channel(artifact: dict) -> None:
    """Reject an artifact whose channel is not the single declared
    deployable constant (no same-slug or diagnostic channel may
    carry a deployable cycles number)."""
    if artifact.get("deployable_channel") != DEPLOYABLE_CHANNEL:
        raise DeployLegalityError(
            "artifact deployable_channel is not the declared "
            "constant")
    for row in artifact.get("cells", []):
        if row.get("status") == "typed":
            continue
        if row.get("prediction_channel") != DEPLOYABLE_CHANNEL:
            raise DeployLegalityError(
                f"cell {row.get('slug')}/{row.get('shape_point')} "
                f"carries channel {row.get('prediction_channel')!r}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--emissions", type=Path, default=DEFAULT_EMISSIONS)
    parser.add_argument(
        "--walker", type=Path, default=DEFAULT_WALKER)
    parser.add_argument(
        "--v5-head", type=Path, default=DEFAULT_V5_HEAD)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument(
        "--witness", type=Path, default=DEFAULT_WITNESS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--schema-path", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    artifact = build_evaluation(
        cohort_path=args.cohort,
        split_path=args.split,
        emissions_path=args.emissions,
        walker_path=args.walker,
        v5_head_path=args.v5_head,
        card_path=args.card,
        witness_path=args.witness,
    )
    validate_artifact_channel(artifact)
    import jsonschema
    schema = json.loads(
        args.schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"gcom_lite_cycles_evaluator: wrote {args.out} "
        f"({artifact['n_modeled']} modeled / "
        f"{artifact['n_typed']} typed; n_fit="
        f"{artifact['fit']['n_fit']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
