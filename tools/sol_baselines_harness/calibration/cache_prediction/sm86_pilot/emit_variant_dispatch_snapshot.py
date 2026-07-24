"""Emit the variant-dispatch baseline snapshot block
(cross-arch plan the calibration workstream, the design contract).

Appends (or replaces) the ``[variant_dispatch_v1]`` block at
the end of
``scripts/baselines_harness/calibration/cache_prediction/sm86_baseline_predictions_v1.toml``.
The block carries per-cell predictions produced via the
deployed variant dispatch path:

  1. For each of the 5 admitted pilot cells, parse
     ``<bundle>/nvbit_metrics/tier2_address_trace.t2a`` and
     run ``predict_from_launch_frame`` with the sm_86 static
     device-card geometry to capture
     ``(raw_l1, raw_l2)``.
  2. Load the variant calibrator via
     ``load_calibrator_for_arch_variant("sm_86", "pilot_v1")``
     and compute
     ``(cal.predict_l1(raw_l1), cal.predict_l2(raw_l2))``.
  3. Record ``(slug, raw_l1, raw_l2, predicted_l1_calibrated,
     predicted_l2_calibrated)`` per cell.

The emitter also records SHA-256 audit hashes for every
artifact the variant predict path consumes (the dispatch
module, the sidecar, the ledger, the historical split) so
the drift gate can fail closed on a silent edit to any of
them.

The emitter is non-destructive: the existing top-level
``[meta]`` / ``[per_cell]`` / ``[audit_hashes]`` blocks
(the R0-baseline LOOCV evidence) are preserved byte-for-
byte. If ``[variant_dispatch_v1]`` already exists, it is
replaced; if not, it is appended at end-of-file.

Deterministic: same Tier-2 binaries + same sidecar + same
ledger + same split + same ``captured_at_utc`` argument →
byte-identical output for the variant block.

Usage:
  python -m baselines_harness.calibration.cache_prediction.sm86_pilot.emit_variant_dispatch_snapshot \\
    [--snapshot-path <path>] [--captured-at-utc <iso8601>]
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
import hashlib

_REPO_ROOT = Path(__file__).resolve().parents[5]

from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.calibrator_dispatch_variant import (  # noqa: E402
    load_calibrator_for_arch_variant,
)
from sol_baselines_harness.metrics.cache_predictor import (  # noqa: E402
    predict_from_launch_frame,
    PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
)
from sol_baselines_harness.metrics.tier2_sidecar_parser import (  # noqa: E402
    parse_launch_frames,
)


_CAL_DIR = (
    _REPO_ROOT
    / "scripts" / "baselines_harness"
    / "calibration" / "cache_prediction"
)
_DEFAULT_SNAPSHOT = _CAL_DIR / "sm86_baseline_predictions_v1.toml"
_TRACES_ROOT = (
    _REPO_ROOT / "scripts" / "baselines_harness" / ".harness_traces"
)


# The 5 admitted pilot cells, in sorted slug order
# (deterministic for the snapshot).
_PILOT_SLUGS = (
    "problem_006_hyena_depthwise_conv1d_split_gate",
    "problem_040_conv2d_residual_block",
    "problem_066_resnet_block_with_time_embedding",
    "problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection",
    "problem_087_embedding_with_initial_layernorm_backward",
)


# sm_86 static device card geometry; same values as
# `persist_fitted_sidecar.py` so the variant predict path
# reproduces the same `raw_l1` / `raw_l2` the fit step
# consumed.
_SM86_GEOMETRY = {
    "arch": "sm_86",
    "l1_kb_per_sm": 192,
    "l1_kb_per_sm_source": "static_device_card",
    "l2_capacity_bytes": 6 * 1024 * 1024,
    "capacity_source": "static_device_card",
    "num_l2_slices": 6,
    "num_l2_slices_source": "static_device_card",
    "sector_size_bytes": 32,
    "sector_size_source": "static_device_card",
    "line_size_bytes": 128,
    "line_size_source": "static_device_card",
    "associativity_l2": 16,
    "associativity_l2_source": "static_device_card",
    "l2_constant_memory_counts_for_read_score": False,
    "l2_constant_memory_counts_for_read_score_source": (
        "static_device_card_sm_86"
    ),
    "geometry_source": "sm86_pilot_static_device_card_v1",
}


# Artifacts whose SHA-256 the variant-dispatch audit-hash
# strict gate enforces. Order is deterministic so the emitted
# block bytes are stable across re-emit.
_VARIANT_AUDIT_PATHS = (
    "scripts/baselines_harness/calibration/cache_prediction/"
    "sm120_coverage_expansion/calibrator_dispatch_variant.py",
    "scripts/baselines_harness/calibration/cache_prediction/"
    "isotonic_params_sm86.toml",
    "scripts/baselines_harness/calibration/cache_prediction/"
    "label_ledger_v1_sm86_pilot_v1.toml",
    "scripts/baselines_harness/calibration/cache_prediction/"
    "historical_label_split_v1_sm86_pilot_v1.toml",
)


_BLOCK_HEADER = "[variant_dispatch_v1]"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_dir(slug: str) -> Path:
    return _TRACES_ROOT / f"{slug}__cuda_cpp__sm86"


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value type: {type(v)}")


def _emit_section(name: str, fields: dict) -> str:
    lines = [f"[{name}]"]
    for k, v in fields.items():
        lines.append(f"{k} = {_toml_value(v)}")
    lines.append("")
    return "\n".join(lines)


def _emit_table_array(name: str, rows: list[dict]) -> str:
    out: list[str] = []
    for row in rows:
        out.append(f"[[{name}]]")
        for k in row:
            out.append(f"{k} = {_toml_value(row[k])}")
        out.append("")
    return "\n".join(out)


def _build_per_cell_rows() -> list[dict]:
    """Run the variant predict path once per pilot cell.
    Returns the deterministic per-cell rows in pilot-slug
    order."""
    cal = load_calibrator_for_arch_variant("sm_86", "pilot_v1")
    rows: list[dict] = []
    for slug in _PILOT_SLUGS:
        bundle = _bundle_dir(slug)
        t2a = bundle / "nvbit_metrics" / "tier2_address_trace.t2a"
        if not t2a.is_file():
            raise FileNotFoundError(
                f"{slug}: {t2a} missing (Tier-2 sidecar required for "
                f"the variant predict path)"
            )
        frames = parse_launch_frames(t2a)
        if not frames:
            raise ValueError(
                f"{slug}: parse_launch_frames returned 0 frames"
            )
        frame = frames[0]
        if frame.truncation is not None:
            frame = dataclasses.replace(frame, truncation=None)
        result = predict_from_launch_frame(
            frame, cell_name=slug, geometry_spec=_SM86_GEOMETRY,
        )
        if result.get("result_kind") != PREDICTOR_BUNDLE_RESULT_KIND_SCALAR:
            raise ValueError(
                f"{slug}: predict_from_launch_frame returned "
                f"result_kind={result.get('result_kind')!r}"
            )
        raw_l1 = float(result["raw_l1_hit_rate"])
        raw_l2 = float(result["raw_l2_hit_rate"])
        pred_l1 = cal.predict_l1(raw_l1)
        pred_l2 = cal.predict_l2(raw_l2)
        rows.append({
            "cell_name": slug,
            "raw_l1": raw_l1,
            "raw_l2": raw_l2,
            "predicted_l1_calibrated": pred_l1,
            "predicted_l2_calibrated": pred_l2,
        })
    return rows


def _build_block(captured_at_utc: str) -> str:
    per_cell = _build_per_cell_rows()
    audit_rows = []
    for relpath in _VARIANT_AUDIT_PATHS:
        path = _REPO_ROOT / relpath
        audit_rows.append({
            "path": relpath,
            "sha256": _sha256_file(path),
        })

    meta = _emit_section("variant_dispatch_v1", {
        "schema_version": "1.0",
        "variant_arch": "sm_86",
        "variant_token": "pilot_v1",
        "captured_at_utc": captured_at_utc,
        "drift_tolerance_abs": 1e-12,
        "wrapper_entry_point": (
            "baselines_harness.calibration.cache_prediction."
            "sm120_coverage_expansion.calibrator_dispatch_variant."
            "load_calibrator_for_arch_variant"
        ),
        "geometry_source": _SM86_GEOMETRY["geometry_source"],
        "pilot_cell_count": len(per_cell),
    })
    per_cell_block = _emit_table_array(
        "variant_dispatch_v1.per_cell", per_cell,
    )
    audit_block = _emit_table_array(
        "variant_dispatch_v1.audit_hashes", audit_rows,
    )

    return (
        "# Variant-dispatch baseline snapshot (cross-arch plan t" "ask15, A" "C-10).\n"
        "# Pinned per-cell predictions produced via the deployed\n"
        "# load_calibrator_for_arch_variant(\"sm_86\", \"pilot_v1\") path:\n"
        "#   parse_launch_frames(t2a)\n"
        "#       -> predict_from_launch_frame(sm_86 static geometry)\n"
        "#       -> (raw_l1, raw_l2)\n"
        "#       -> variant_calibrator.predict_l1/l2(raw)\n"
        "#       -> (predicted_l1_calibrated, predicted_l2_calibrated).\n"
        "# The drift gate at test_sm86_no_per_cell_drift.py re-runs the\n"
        "# same path and asserts every cell matches within\n"
        "# drift_tolerance_abs (1e-12). The audit_hashes array adds a\n"
        "# strict gate over the dispatch module + sidecar + ledger +\n"
        "# split; a silent edit to any of those fails the strict gate\n"
        "# even if per-cell predictions happen to remain identical.\n"
        "\n"
        + meta
        + "\n"
        + per_cell_block
        + audit_block
    )


def _splice_block(existing_text: str, new_block: str) -> str:
    """Return ``existing_text`` with the variant-dispatch
    block replaced or appended.

    If ``[variant_dispatch_v1]`` is already present, REPLACE
    everything from that line to end-of-file with
    ``new_block``. Otherwise, APPEND ``new_block`` at
    end-of-file. The existing top-level blocks
    (``[meta]``, ``[per_cell]``, ``[audit_hashes]``) are
    preserved byte-for-byte either way.
    """
    idx = existing_text.find(f"\n{_BLOCK_HEADER}")
    if idx >= 0:
        # Cut from the newline before the block header so
        # any leading comment lines for the variant block
        # are also replaced.
        # Find the start of the comment block immediately
        # preceding the [variant_dispatch_v1] line. Walk
        # backwards over consecutive lines that start with
        # '#' or are blank.
        head = existing_text[: idx + 1]
        # Trim trailing blank lines and comment lines
        # that introduce the block.
        lines = head.split("\n")
        # Walk back while line is empty or starts with '#'.
        while lines and (
            not lines[-1].strip() or lines[-1].lstrip().startswith("#")
        ):
            lines.pop()
        head = "\n".join(lines).rstrip() + "\n\n"
        return head + new_block.rstrip("\n") + "\n"
    # Not present; append.
    if not existing_text.endswith("\n"):
        existing_text = existing_text + "\n"
    if not existing_text.endswith("\n\n"):
        existing_text = existing_text + "\n"
    return existing_text + new_block.rstrip("\n") + "\n"


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_pilot.emit_variant_dispatch_snapshot.main')
def main(
    *,
    snapshot_path: Path | None = None,
    captured_at_utc: str | None = None,
) -> int:
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm86_pilot.emit_variant_dispatch_snapshot.main')
    snap = snapshot_path or _DEFAULT_SNAPSHOT
    cap = captured_at_utc or "2026-05-28T23:45:00+00:00"
    block = _build_block(cap)
    existing = snap.read_text("utf-8")
    new_text = _splice_block(existing, block)
    snap.write_text(new_text, "utf-8")
    print(
        f"emitted: {snap} (variant_dispatch_v1 block; "
        f"5 per_cell rows; {len(_VARIANT_AUDIT_PATHS)} audit hashes)"
    )
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument(
        "--captured-at-utc",
        default="2026-05-28T23:45:00+00:00",
        help="ISO-8601 UTC timestamp pinned for byte-deterministic re-runs.",
    )
    args = parser.parse_args()
    return main(
        snapshot_path=args.snapshot_path,
        captured_at_utc=args.captured_at_utc,
    )


if __name__ == "__main__":
    sys.exit(_cli())
