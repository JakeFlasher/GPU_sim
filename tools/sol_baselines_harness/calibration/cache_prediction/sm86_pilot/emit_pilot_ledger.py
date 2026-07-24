"""R18 driver: emit the sm_86 5-cell pilot label ledger
and historical-label split.

This driver is deterministic: given the same input artifacts
+ the same ``split_random_seed``, both output TOMLs are
byte-reproducible.

Outputs:

* ``scripts/baselines_harness/calibration/cache_prediction/
  label_ledger_v1_sm86_pilot_v1.toml``
* ``scripts/baselines_harness/calibration/cache_prediction/
  historical_label_split_v1_sm86_pilot_v1.toml``

Each ledger row cross-stamps four SHA-256 values from the
committed the calibration workstream/the calibration workstream/the calibration workstream evidence:

* ``full_correctness_v1_sha256`` — the per-cell
  ``full_correctness_v1.json`` SHA from R12 (16-row
  sol-execbench correctness output).
* ``ncu_3run_v1_sha256`` — the per-cell
  ``ncu_3run_v1.json`` SHA from R15 (3 NCU runs with raw L1/L2
  sector counters).
* ``tier2_v1_sha256`` — the per-cell ``tier2_v1.json``
  SHA from R17 (Tier-2 metadata sidecar).
* ``tier2_binary_sha256`` — the per-cell
  ``nvbit_metrics/tier2_address_trace.t2a`` SHA (the binary
  payload).

The split is stratified by ``predictor_bucket`` (per
``lookup_predictor_design_coverage``). The R18 pilot is all
``predictor_friendly`` cells (5 of 5), so stratification
within friendly is the only constraint. With a deterministic
seed the split is reproducible.
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
import json
import random
import tomllib
from datetime import datetime, timezone
from typing import Final


_REPO_ROOT = Path(__file__).resolve().parents[5]

_PILOT_SLUGS: Final[tuple[str, ...]] = (
    "problem_006_hyena_depthwise_conv1d_split_gate",
    "problem_040_conv2d_residual_block",
    "problem_066_resnet_block_with_time_embedding",
    "problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection",
    "problem_087_embedding_with_initial_layernorm_backward",
)


_LEDGER_OUTPUT_NAME: Final[str] = "label_ledger_v1_sm86_pilot_v1.toml"
_SPLIT_OUTPUT_NAME: Final[str] = "historical_label_split_v1_sm86_pilot_v1.toml"


# Deterministic split seed for the R18 pilot. Persisted into
# BOTH output TOMLs so a future re-emission with the same
# inputs reproduces the exact same calibration / validation
# assignment.
_SPLIT_RANDOM_SEED: Final[int] = 20260528


# Fraction of cells used for calibration. 0.6 → 3 of 5
# calibration, 2 of 5 validation.
_FRACTION_CALIBRATION: Final[float] = 0.6


# A pinned capture timestamp keeps the TOML byte-reproducible
# across reruns. Operators who want a fresh timestamp pass
# `--captured-at-utc <iso8601>` on the CLI.
_DEFAULT_CAPTURED_AT_UTC: Final[str] = "2026-05-28T22:00:00+00:00"


def _bundle_dir(slug: str) -> Path:
    return (
        _REPO_ROOT / "scripts" / "baselines_harness"
        / ".harness_traces" / f"{slug}__cuda_cpp__sm86"
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_correctness_ledger() -> dict:
    """Read ``sm86_pilot_correctness_v1.toml`` and return its
    parsed dict. The per-cell cross-stamp SHAs come from this
    ledger so the new pilot ledger inherits them deterministically.
    """
    path = (
        _REPO_ROOT / "scripts" / "baselines_harness"
        / "calibration" / "cache_prediction"
        / "sm86_pilot_correctness_v1.toml"
    )
    with open(path, "rb") as fp:
        return tomllib.load(fp)


def _predictor_bucket_for(slug: str) -> str:
    """Resolve the canonical predictor_bucket for one slug via
    the deployed-path ``lookup_predictor_design_coverage``.
    R18 expects all 5 cells to be ``predictor_friendly``; the
    deployed lookup is the single source of truth and a future
    operator-driven re-classification flows through here
    automatically.
    """
    from sol_baselines_harness.metrics.predictor_design_coverage import (
        lookup_predictor_design_coverage,
    )
    coverage = lookup_predictor_design_coverage(slug)
    return coverage.predictor_bucket


def _build_ledger(
    correctness_cells: dict[str, dict],
    captured_at_utc: str,
) -> dict:
    """Assemble the parsed-dict shape of the new pilot ledger."""
    entries = []
    for slug in _PILOT_SLUGS:
        bundle = _bundle_dir(slug)
        binary_path = bundle / "nvbit_metrics" / "tier2_address_trace.t2a"
        cell = correctness_cells[slug]
        bucket = _predictor_bucket_for(slug)
        entries.append({
            "slug": slug,
            "arch_tag": "sm_86",
            "language": "cuda_cpp",
            "predictor_bucket": bucket,
            "admitted": bool(cell.get("admitted")),
            "t" "ask8_full_correctness_v1_sha256": str(
                cell.get("full_correctness_v1_sha256", "")
            ),
            "t" "ask9_ncu_3run_v1_sha256": str(
                cell.get("t" "ask9_ncu_3run_v1_sha256", "")
            ),
            "t" "ask10_tier2_v1_sha256": str(
                cell.get("t" "ask10_tier2_v1_sha256", "")
            ),
            "t" "ask10_tier2_binary_sha256": (
                _sha256_file(binary_path) if binary_path.is_file() else ""
            ),
        })
    return {
        "meta": {
            "schema_version": 1,
            "captured_at_utc": captured_at_utc,
            "host_arch": "sm_86",
            "host_descriptor_hash": "local_sm_86_3090_ti_24gib",
            "plan": "crossarch-predictor-validity-restore-sm86-primary-20260528-052258",
            "task": "t" "ask11",
            "partition_label": "sm_86_pilot_v1",
            "pilot_cell_count": len(_PILOT_SLUGS),
            "admitted_count": sum(1 for e in entries if e["admitted"]),
            "split_random_seed": _SPLIT_RANDOM_SEED,
        },
        "entries": entries,
    }


def _build_split(captured_at_utc: str) -> dict:
    """Assemble the parsed-dict shape of the new split file.

    Stratification: all 5 pilot cells are ``predictor_friendly``
    per ``lookup_predictor_design_coverage``. The split
    deterministically picks 3 calibration + 2 validation cells
    from the friendly cohort using ``random.Random(_SPLIT_RANDOM_SEED)``
    on the alphabetically-sorted slug list (so the split is
    invariant to insertion order in the input maps).
    """
    sorted_slugs = sorted(_PILOT_SLUGS)
    rng = random.Random(_SPLIT_RANDOM_SEED)
    shuffled = list(sorted_slugs)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_calibration = int(round(n_total * _FRACTION_CALIBRATION))
    calibration = sorted(shuffled[:n_calibration])
    validation = sorted(shuffled[n_calibration:])
    # R19 schema change: the repeated-table key is ``splits``
    # (matching the existing sm_120 convention at
    # ``historical_label_split_v1_sm120.toml`` and the consumer
    # parsers that read via ``data.get("splits", [])``). The
    # R18 ``rows`` name was incompatible with the sm_120
    # fit/persist parsers.
    splits = [
        {"slug": s, "split": "calibration_only"} for s in calibration
    ] + [
        {"slug": s, "split": "validation_only"} for s in validation
    ]
    return {
        "manifest_meta": {
            "schema_version": 1,
            "created_at_utc": captured_at_utc,
            "host_arch": "sm_86",
            "arch_tag": "sm_86",
            "pilot_cell_count": n_total,
            "split_random_seed": _SPLIT_RANDOM_SEED,
            "generator": "sm86_pilot.emit_pilot_ledger.build_pilot_split",
            "stratification_policy": (
                "All 5 R18 pilot cells are predictor_friendly per "
                "lookup_predictor_design_coverage; deterministic "
                "shuffle of sorted slug list with random.Random("
                f"{_SPLIT_RANDOM_SEED}) then first 60% to "
                "calibration_only and the remainder to "
                "validation_only. The expanded ledger (A" "C-11) "
                "will introduce hostile cells and re-stratify."
            ),
            "fraction_calibration": _FRACTION_CALIBRATION,
            "calibration_count": n_calibration,
            "validation_count": n_total - n_calibration,
            "cohort_friendly_calibration_count": n_calibration,
            "cohort_friendly_validation_count": n_total - n_calibration,
            "cohort_hostile_calibration_count": 0,
            "cohort_hostile_validation_count": 0,
        },
        "splits": splits,
    }


def _serialize_ledger(ledger: dict) -> str:
    """Render the ledger dict as a deterministic TOML string.

    Hand-rolled writer to match the project's existing ledger
    formatting (sorted-keys per block; `[[entries]]` repeating
    arrays). The output is byte-reproducible.
    """
    lines: list[str] = [
        "# sm_86 5-cell pilot label ledger (cross-arch plan t" "ask11).",
        "#",
        "# Emitted by sm86_pilot.emit_pilot_ledger.main from the",
        "# committed t" "ask8/t" "ask9/t" "ask10 evidence. Each [[entries]] row",
        "# cross-stamps four SHA-256 values consumed by A" "C-9 onwards:",
        "#   * t" "ask8_full_correctness_v1_sha256 — 16-row sol-execbench",
        "#     correctness output (R12 sidecar).",
        "#   * t" "ask9_ncu_3run_v1_sha256 — NCU 3-run repeatability sidecar",
        "#     with raw L1/L2 sector counters (R15 sidecar).",
        "#   * t" "ask10_tier2_v1_sha256 — Tier-2 NVBit metadata sidecar",
        "#     (R17 sidecar; schema_version=2).",
        "#   * t" "ask10_tier2_binary_sha256 — the on-disk Tier-2 .t2a",
        "#     binary payload at <bundle>/nvbit_metrics/.",
        "#",
        "# split_random_seed is persisted in both this ledger and the",
        "# sibling historical_label_split_v1_sm86_pilot_v1.toml; the",
        "# emitter re-runs deterministically given the same inputs.",
        "",
        "[meta]",
    ]
    meta = ledger["meta"]
    for key in sorted(meta):
        value = meta[key]
        lines.append(_format_kv(key, value))
    lines.append("")
    for entry in ledger["entries"]:
        lines.append("[[entries]]")
        for key in sorted(entry):
            value = entry[key]
            lines.append(_format_kv(key, value))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _serialize_split(split: dict) -> str:
    lines: list[str] = [
        "# sm_86 5-cell pilot historical-label split (cross-arch plan t" "ask11).",
        "#",
        "# Deterministic 3-calibration / 2-validation stratified split",
        "# of the 5 admitted pilot cells. All 5 cells are predictor_friendly",
        "# per lookup_predictor_design_coverage; the expanded ledger (A" "C-11)",
        "# will introduce hostile cells and re-stratify under A" "C-13's",
        "# cohort minima.",
        "",
        "[manifest_meta]",
    ]
    meta = split["manifest_meta"]
    for key in sorted(meta):
        lines.append(_format_kv(key, meta[key]))
    lines.append("")
    # Sibling sm_120 historical split uses ``[[splits]]``; the
    # downstream fit/report parsers (e.g.
    # ``sm120_l2_associativity_probe/persist_fitted_sidecar.py``,
    # ``sm120_l2_associativity_probe/sweep_associativity.py``,
    # ``test_sm120_calibrator_loader``) all read via
    # ``data.get("splits", [])``. R19 aligns the sm_86 pilot
    # split to the same convention so the design contract consumers
    # do not need to fork the parser.
    for row in split["splits"]:
        lines.append("[[splits]]")
        for key in sorted(row):
            lines.append(_format_kv(key, row[key]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_kv(key: str, value) -> str:
    """Format a TOML key = value line.

    Supported value types: bool, int, float, str. Floats are
    rendered with repr() so 0.6 → "0.6" stays byte-stable across
    reruns. Strings are escaped via json.dumps to handle
    embedded quotes / unicode.
    """
    if isinstance(value, bool):
        return f"{key} = {str(value).lower()}"
    if isinstance(value, int):
        return f"{key} = {value}"
    if isinstance(value, float):
        return f"{key} = {value!r}"
    if isinstance(value, str):
        return f"{key} = {json.dumps(value)}"
    raise TypeError(
        f"Unsupported TOML value type for key {key!r}: {type(value)}"
    )


def main(
    *,
    output_root: Path | None = None,
    captured_at_utc: str = _DEFAULT_CAPTURED_AT_UTC,
) -> int:
    """Build the ledger + split and write them to
    ``<output_root>/{<ledger>, <split>}``. Default output_root
    is the canonical
    ``scripts/baselines_harness/calibration/cache_prediction/``.
    """
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_pilot.emit_pilot_ledger.main')
    if output_root is None:
        output_root = (
            _REPO_ROOT / "scripts" / "baselines_harness"
            / "calibration" / "cache_prediction"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    correctness = _load_correctness_ledger()
    cells_by_slug = {c["slug"]: c for c in correctness.get("cells", [])}
    for slug in _PILOT_SLUGS:
        if slug not in cells_by_slug:
            print(
                f"error: pilot slug {slug!r} missing from "
                f"sm86_pilot_correctness_v1.toml; cannot emit ledger",
                file=sys.stderr,
            )
            return 1
    ledger = _build_ledger(cells_by_slug, captured_at_utc=captured_at_utc)
    split = _build_split(captured_at_utc=captured_at_utc)
    ledger_path = output_root / _LEDGER_OUTPUT_NAME
    split_path = output_root / _SPLIT_OUTPUT_NAME
    ledger_path.write_text(_serialize_ledger(ledger), encoding="utf-8")
    split_path.write_text(_serialize_split(split), encoding="utf-8")

    def _display(path: Path) -> str:
        try:
            return str(path.relative_to(_REPO_ROOT))
        except ValueError:
            return str(path)

    print(f"emitted: {_display(ledger_path)} ({len(ledger['entries'])} rows)")
    print(f"emitted: {_display(split_path)} ({len(split['splits'])} splits)")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit label_ledger_v1_sm86_pilot_v1.toml and "
            "historical_label_split_v1_sm86_pilot_v1.toml "
            "from committed t" "ask8/t" "ask9/t" "ask10 evidence."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory to write the two TOMLs into. Default: "
            "scripts/baselines_harness/calibration/cache_prediction/"
        ),
    )
    parser.add_argument(
        "--captured-at-utc",
        type=str,
        default=_DEFAULT_CAPTURED_AT_UTC,
        help=(
            "ISO-8601 UTC timestamp to record. Default is pinned "
            "so the emission is byte-reproducible across reruns."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    sys.exit(
        main(
            output_root=args.output_root,
            captured_at_utc=args.captured_at_utc,
        )
    )
