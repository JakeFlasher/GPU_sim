"""Drive the simulator + deployed sm_120 calibrator
over the pilot cohort's Tier-2 traces.

Reads the upstream pilot status artifacts:
  * ``sm120_pilot_tier2_harvest_status_v1.toml`` —
    per-cell sidecar SHA + record/launch counts.
  * ``sm120_pilot_ncu_repeatability_v1.toml`` —
    per-cell L1/L2 NCU labels (only completed rows
    contribute the ground-truth side of the
    predictor-vs-ncu comparison).

For each pilot that completed both upstream gates:
  * Verifies the sidecar exists on disk + the file
    SHA-256 matches the recorded SHA (typed
    ``predictor_sidecar_sha_mismatch`` on drift).
  * Parses the sidecar via
    ``tier2_sidecar_parser.parse_launch_frames``.
  * Resolves sm_120 cache geometry once.
  * Per-frame, calls
    ``cache_predictor.predict_from_launch_frame`` to
    get raw L1/L2 hit rates (averaged across frames
    weighted by sector counts when multiple).
  * Loads the deployed sm_120 calibrator
    (``load_calibrator_for_arch("sm_120")``) once.
  * Calls ``predict_l1`` / ``predict_l2`` on the
    averaged raw values to produce calibrated
    predictions.

Pilots that didn't complete the NCU gate
(ATen-dispatched cells with no stable per-kernel
name to filter NCU on) get the typed
``predictor_pilot_not_in_ncu_status`` failure
BEFORE any predictor work — they are not admissible
for the per-kernel predictor comparison.

The emitted artifact carries one row per pilot
(12 rows total) with per-cell raw + calibrated +
ncu evidence so the next-round diagnostics refresh
and expanded ledger can ingest without re-running
the simulator.
"""

from __future__ import annotations

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import dataclasses
import hashlib
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Final


_REPO_ROOT = Path(__file__).resolve().parents[5]
_CACHE_PREDICTION_DIR = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
)
_DEFAULT_NCU_STATUS_PATH = (
    _CACHE_PREDICTION_DIR
    / "sm120_pilot_ncu_repeatability_v1.toml"
)
_DEFAULT_TIER2_STATUS_PATH = (
    _CACHE_PREDICTION_DIR
    / "sm120_pilot_tier2_harvest_status_v1.toml"
)
_DEFAULT_HARVEST_OUTPUT_ROOT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / ".harness_traces"
)


PREDICTOR_PREDICTIONS_SCHEMA_VERSION: Final[str] = "1.0"
PREDICTOR_PREDICTIONS_HOST_ARCH: Final[str] = "sm_120"

# sm_120 GB206 L2 simulator-effective capacity (30 MiB).
# Physical L2 is 32 MiB, but the simulator's set-count
# invariant ``(l2_capacity_bytes // num_l2_slices) %
# (line_size_bytes * associativity_l2) == 0`` requires
# a divisor-friendly value: 32 MiB / 24 slices does
# NOT divide cleanly into the 128B*32-way 4096-byte
# set granularity; 30 MiB / 24 slices = 1280 KiB per
# slice does. The offline-calibration rounding is
# documented in BL-20260526-empirical-geometry-sweep-
# with-trace-replay.
_SM120_L2_CAPACITY_BYTES: Final[int] = 30 * 1024 * 1024

# Sentinels for NULL float / int fields on typed-failure
# rows. Match the diagnostics sidecar's NULL_FLOAT /
# NULL_INT conventions so the downstream consumer doesn't
# need a separate vocabulary.
_NULL_FLOAT: Final[float] = -1.0
_NULL_SHA: Final[str] = ""


# Closed vocabulary of failure-reason tokens. Empty string
# is the implicit "no failure" value when
# ``predictor_success=True``.
PREDICTOR_PREDICTIONS_FAILURE_VOCABULARY: Final[tuple[str, ...]] = (
    "predictor_pilot_not_in_ncu_status",
    "predictor_pilot_not_in_tier2_status",
    "predictor_sidecar_missing_on_disk",
    "predictor_sidecar_sha_mismatch",
    "predictor_sidecar_parse_failed",
    "predictor_simulator_failed",
    "predictor_calibrator_failed",
)


@dataclass(frozen=True, slots=True)
class PredictorPredictionRow:
    slug: str
    language: str
    sidecar_sha256: str
    raw_l1: float
    raw_l2: float
    predicted_l1: float
    predicted_l2: float
    ncu_l1: float
    ncu_l2: float
    predictor_success: bool
    failure_reason: str
    captured_at_utc: str


class PredictorPredictionsOutcomeInvalidError(ValueError):
    """Raised when a row's failure_reason is outside the
    closed vocabulary."""


def _validate_failure_reason(reason: str) -> None:
    if reason and reason not in PREDICTOR_PREDICTIONS_FAILURE_VOCABULARY:
        raise PredictorPredictionsOutcomeInvalidError(
            f"failure_reason={reason!r} not in "
            f"PREDICTOR_PREDICTIONS_FAILURE_VOCABULARY "
            f"{sorted(PREDICTOR_PREDICTIONS_FAILURE_VOCABULARY)}"
        )


def _build_failure_row(
    *,
    slug: str,
    language: str,
    sidecar_sha256: str,
    failure_reason: str,
    ncu_l1: float,
    ncu_l2: float,
    timestamp: str,
) -> PredictorPredictionRow:
    _validate_failure_reason(failure_reason)
    return PredictorPredictionRow(
        slug=slug,
        language=language,
        sidecar_sha256=sidecar_sha256,
        raw_l1=_NULL_FLOAT,
        raw_l2=_NULL_FLOAT,
        predicted_l1=_NULL_FLOAT,
        predicted_l2=_NULL_FLOAT,
        ncu_l1=ncu_l1,
        ncu_l2=ncu_l2,
        predictor_success=False,
        failure_reason=failure_reason,
        captured_at_utc=timestamp,
    )


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ncu_completed_map(
    ncu_status_path: Path,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Return {(slug, language): (ncu_l1_mean, ncu_l2_mean)}
    for every completed NCU row."""
    data = tomllib.loads(ncu_status_path.read_text("utf-8"))
    result: dict[tuple[str, str], tuple[float, float]] = {}
    for entry in data.get("entries", []):
        if not entry.get("completed"):
            continue
        result[(entry["slug"], entry["language"])] = (
            float(entry["ncu_l1_mean"]),
            float(entry["ncu_l2_mean"]),
        )
    return result


def _load_tier2_completed_map(
    tier2_status_path: Path,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Return {(slug, language): (sidecar_relpath, sidecar_sha256)}
    for every completed Tier-2 row."""
    data = tomllib.loads(tier2_status_path.read_text("utf-8"))
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for entry in data.get("entries", []):
        if not entry.get("completed"):
            continue
        result[(entry["slug"], entry["language"])] = (
            entry["sidecar_relpath"],
            entry["tier2_sidecar_sha256"],
        )
    return result


def _drive_one_pilot(
    *,
    slug: str,
    language: str,
    sidecar_path: Path,
    expected_sha256: str,
    ncu_l1: float,
    ncu_l2: float,
    calibrator,
    geometry_spec,
    parse_fn: Callable,
    timestamp: str,
) -> PredictorPredictionRow:
    """Drive the simulator + calibrator over one pilot
    cell's Tier-2 sidecar. Returns a typed-failure row
    on any pipeline step that raises."""
    # SHA verification.
    if not sidecar_path.is_file():
        return _build_failure_row(
            slug=slug,
            language=language,
            sidecar_sha256=_NULL_SHA,
            failure_reason="predictor_sidecar_missing_on_disk",
            ncu_l1=ncu_l1,
            ncu_l2=ncu_l2,
            timestamp=timestamp,
        )
    live_sha = _sha256_of_file(sidecar_path)
    if live_sha != expected_sha256:
        return _build_failure_row(
            slug=slug,
            language=language,
            sidecar_sha256=live_sha,
            failure_reason="predictor_sidecar_sha_mismatch",
            ncu_l1=ncu_l1,
            ncu_l2=ncu_l2,
            timestamp=timestamp,
        )

    # Parse sidecar.
    try:
        frames = parse_fn(sidecar_path)
    except Exception:  # noqa: BLE001
        return _build_failure_row(
            slug=slug,
            language=language,
            sidecar_sha256=live_sha,
            failure_reason="predictor_sidecar_parse_failed",
            ncu_l1=ncu_l1,
            ncu_l2=ncu_l2,
            timestamp=timestamp,
        )
    if not frames:
        return _build_failure_row(
            slug=slug,
            language=language,
            sidecar_sha256=live_sha,
            failure_reason="predictor_sidecar_parse_failed",
            ncu_l1=ncu_l1,
            ncu_l2=ncu_l2,
            timestamp=timestamp,
        )

    # Drive simulator over each launch frame; aggregate
    # raw L1/L2 by sector-weighted average.
    from sol_baselines_harness.metrics.cache_predictor import (
        predict_from_launch_frame,
    )

    l1_hits = 0
    l1_total = 0
    l2_hits = 0
    l2_total = 0
    try:
        for frame in frames:
            # Offline-calibration truncation-relaxation per
            # BL-20260526-empirical-geometry-sweep-with-trace-replay:
            # strip the truncation marker before driving the simulator
            # so the predictor consumes the captured ~524k-record
            # prefix instead of abstaining with
            # ``trace_budget_exceeded``. The truncation provenance is
            # preserved upstream in
            # ``sm120_pilot_tier2_harvest_status_v1.toml``'s
            # ``truncation_relaxation_applied=true`` row for the
            # cell; downstream MAPE attribution interprets the
            # prediction as "raw hit-rate on the captured prefix".
            if getattr(frame, "truncation", None) is not None:
                frame = dataclasses.replace(frame, truncation=None)
            result = predict_from_launch_frame(
                frame, cell_name=slug, geometry_spec=geometry_spec,
            )
            if result.get("result_kind") != "scalar":
                continue
            l1_hits += int(result["l1_hit_count"])
            l1_total += int(result["l1_total_count"])
            l2_hits += int(result["l2_hit_count"])
            l2_total += int(result["l2_total_count"])
    except Exception:  # noqa: BLE001
        return _build_failure_row(
            slug=slug,
            language=language,
            sidecar_sha256=live_sha,
            failure_reason="predictor_simulator_failed",
            ncu_l1=ncu_l1,
            ncu_l2=ncu_l2,
            timestamp=timestamp,
        )
    if l1_total == 0 or l2_total == 0:
        return _build_failure_row(
            slug=slug,
            language=language,
            sidecar_sha256=live_sha,
            failure_reason="predictor_simulator_failed",
            ncu_l1=ncu_l1,
            ncu_l2=ncu_l2,
            timestamp=timestamp,
        )
    raw_l1 = l1_hits / l1_total
    raw_l2 = l2_hits / l2_total

    # Calibrate.
    try:
        predicted_l1 = float(calibrator.predict_l1(raw_l1))
        predicted_l2 = float(calibrator.predict_l2(raw_l2))
    except Exception:  # noqa: BLE001
        return _build_failure_row(
            slug=slug,
            language=language,
            sidecar_sha256=live_sha,
            failure_reason="predictor_calibrator_failed",
            ncu_l1=ncu_l1,
            ncu_l2=ncu_l2,
            timestamp=timestamp,
        )

    return PredictorPredictionRow(
        slug=slug,
        language=language,
        sidecar_sha256=live_sha,
        raw_l1=raw_l1,
        raw_l2=raw_l2,
        predicted_l1=predicted_l1,
        predicted_l2=predicted_l2,
        ncu_l1=ncu_l1,
        ncu_l2=ncu_l2,
        predictor_success=True,
        failure_reason="",
        captured_at_utc=timestamp,
    )


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.drive_pilot_predictor.drive_pilot_predictor_cohort')
def drive_pilot_predictor_cohort(
    pilot_cells: list[tuple[str, str]],
    *,
    ncu_status_path: Path = _DEFAULT_NCU_STATUS_PATH,
    tier2_status_path: Path = _DEFAULT_TIER2_STATUS_PATH,
    harvest_output_root: Path = _DEFAULT_HARVEST_OUTPUT_ROOT,
    captured_at_utc: str | None = None,
    calibrator_loader: Callable | None = None,
    geometry_resolver: Callable | None = None,
    parse_fn: Callable | None = None,
) -> list[PredictorPredictionRow]:
    """Drive the predictor over a pilot cohort.

    Per-cell semantics:

    1. If the pilot is NOT in the NCU status's completed
       set, demote to ``predictor_pilot_not_in_ncu_status``
       BEFORE any pipeline work.
    2. If the pilot is NOT in the Tier-2 status's
       completed set, demote to
       ``predictor_pilot_not_in_tier2_status``.
    3. Otherwise, run the full pipeline (SHA verify →
       parse → simulator → calibrator).

    The ``calibrator_loader``, ``geometry_resolver``,
    and ``parse_fn`` parameters are dependency-injection
    seams: when None, the live production paths are
    used. Tests pass synthetic stubs to exercise the
    contract without spawning the simulator / calibrator
    on real fixtures.

    Returns ASCII-sorted rows.
    """
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.drive_pilot_predictor.drive_pilot_predictor_cohort')
    timestamp = captured_at_utc or datetime.now(
        timezone.utc
    ).replace(microsecond=0).isoformat()

    # Load upstream status maps.
    ncu_completed = _load_ncu_completed_map(ncu_status_path)
    tier2_completed = _load_tier2_completed_map(tier2_status_path)

    if not pilot_cells:
        return []

    # Lazy load calibrator / geometry once for the whole
    # cohort; only when at least one pilot is in both
    # upstream completed sets.
    eligible = [
        (s, l) for (s, l) in pilot_cells
        if (s, l) in ncu_completed and (s, l) in tier2_completed
    ]

    calibrator = None
    geometry_spec = None
    rows: list[PredictorPredictionRow] = []

    if eligible:
        if calibrator_loader is None:
            from sol_baselines_harness.metrics.cache_predictor import (
                load_calibrator_for_arch,
            )
            calibrator_loader = lambda: load_calibrator_for_arch(  # noqa: E731
                "sm_120",
            )
        if geometry_resolver is None:
            from sol_baselines_harness.metrics.cache_geometry import (
                resolve_cache_geometry,
            )
            geometry_resolver = lambda: resolve_cache_geometry(  # noqa: E731
                arch="sm_120",
                cuda_l2_cache_size_bytes=_SM120_L2_CAPACITY_BYTES,
            )[0]
        if parse_fn is None:
            from sol_baselines_harness.metrics.tier2_sidecar_parser import (
                parse_launch_frames,
            )
            parse_fn = parse_launch_frames
        try:
            calibrator = calibrator_loader()
            geometry_spec = geometry_resolver()
        except Exception:  # noqa: BLE001
            # Loader / geometry failure poisons EVERY
            # eligible pilot; emit them as
            # predictor_calibrator_failed.
            calibrator = None

    for slug, language in pilot_cells:
        if (slug, language) not in ncu_completed:
            rows.append(_build_failure_row(
                slug=slug,
                language=language,
                sidecar_sha256=_NULL_SHA,
                failure_reason="predictor_pilot_not_in_ncu_status",
                ncu_l1=_NULL_FLOAT,
                ncu_l2=_NULL_FLOAT,
                timestamp=timestamp,
            ))
            continue
        if (slug, language) not in tier2_completed:
            ncu_l1, ncu_l2 = ncu_completed[(slug, language)]
            rows.append(_build_failure_row(
                slug=slug,
                language=language,
                sidecar_sha256=_NULL_SHA,
                failure_reason="predictor_pilot_not_in_tier2_status",
                ncu_l1=ncu_l1,
                ncu_l2=ncu_l2,
                timestamp=timestamp,
            ))
            continue
        ncu_l1, ncu_l2 = ncu_completed[(slug, language)]
        sidecar_relpath, expected_sha = tier2_completed[
            (slug, language)
        ]
        sidecar_path = _REPO_ROOT / sidecar_relpath
        if calibrator is None:
            rows.append(_build_failure_row(
                slug=slug,
                language=language,
                sidecar_sha256=_NULL_SHA,
                failure_reason="predictor_calibrator_failed",
                ncu_l1=ncu_l1,
                ncu_l2=ncu_l2,
                timestamp=timestamp,
            ))
            continue
        rows.append(_drive_one_pilot(
            slug=slug,
            language=language,
            sidecar_path=sidecar_path,
            expected_sha256=expected_sha,
            ncu_l1=ncu_l1,
            ncu_l2=ncu_l2,
            calibrator=calibrator,
            geometry_spec=geometry_spec,
            parse_fn=parse_fn,
            timestamp=timestamp,
        ))
    rows.sort(key=lambda r: (r.slug, r.language))
    return rows


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_float(value: float) -> str:
    return f"{value!r}"


def render_predictor_predictions_toml(
    *,
    captured_at_utc: str,
    rows: list[PredictorPredictionRow],
) -> str:
    """Render the per-cohort predictor-predictions
    status TOML."""
    aggregate_predictions_emitted = sum(
        1 for r in rows if r.predictor_success
    )
    aggregate_typed_unavailable = sum(
        1 for r in rows if not r.predictor_success
    )
    lines: list[str] = []
    lines.append(
        "# sm_120 pilot predictor-predictions artifact."
    )
    lines.append("#")
    lines.append(
        "# Per-cell raw + calibrated L1/L2 hit rates from"
    )
    lines.append(
        "# the simulator + deployed sm_120 calibrator drive over"
    )
    lines.append(
        "# the pilot Tier-2 sidecars. NCU labels carried"
    )
    lines.append(
        "# as ground-truth-side evidence for the next-round"
    )
    lines.append("# residual computation.")
    lines.append("")
    lines.append("[meta]")
    lines.append(
        f'schema_version = "{PREDICTOR_PREDICTIONS_SCHEMA_VERSION}"'
    )
    lines.append(
        f'host_arch = "{PREDICTOR_PREDICTIONS_HOST_ARCH}"'
    )
    lines.append(f'captured_at_utc = "{captured_at_utc}"')
    lines.append(f"row_count = {len(rows)}")
    lines.append(
        f"aggregate_predictions_emitted = "
        f"{aggregate_predictions_emitted}"
    )
    lines.append(
        f"aggregate_typed_unavailable = "
        f"{aggregate_typed_unavailable}"
    )
    lines.append("")
    for row in rows:
        lines.append("[[entries]]")
        lines.append(f'slug = "{_toml_escape(row.slug)}"')
        lines.append(f'language = "{_toml_escape(row.language)}"')
        lines.append(
            f'sidecar_sha256 = "{_toml_escape(row.sidecar_sha256)}"'
        )
        lines.append(f"raw_l1 = {_format_float(row.raw_l1)}")
        lines.append(f"raw_l2 = {_format_float(row.raw_l2)}")
        lines.append(
            f"predicted_l1 = {_format_float(row.predicted_l1)}"
        )
        lines.append(
            f"predicted_l2 = {_format_float(row.predicted_l2)}"
        )
        lines.append(f"ncu_l1 = {_format_float(row.ncu_l1)}")
        lines.append(f"ncu_l2 = {_format_float(row.ncu_l2)}")
        lines.append(
            f"predictor_success = "
            f"{str(row.predictor_success).lower()}"
        )
        lines.append(
            f'failure_reason = "{_toml_escape(row.failure_reason)}"'
        )
        lines.append(
            f'captured_at_utc = "{row.captured_at_utc}"'
        )
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.drive_pilot_predictor.emit_predictor_predictions')
def emit_predictor_predictions(
    pilot_cells: list[tuple[str, str]],
    *,
    output: Path,
    captured_at_utc: str,
    ncu_status_path: Path = _DEFAULT_NCU_STATUS_PATH,
    tier2_status_path: Path = _DEFAULT_TIER2_STATUS_PATH,
    harvest_output_root: Path = _DEFAULT_HARVEST_OUTPUT_ROOT,
    calibrator_loader: Callable | None = None,
    geometry_resolver: Callable | None = None,
    parse_fn: Callable | None = None,
) -> Path:
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.drive_pilot_predictor.emit_predictor_predictions')
    rows = drive_pilot_predictor_cohort(
        pilot_cells,
        ncu_status_path=ncu_status_path,
        tier2_status_path=tier2_status_path,
        harvest_output_root=harvest_output_root,
        captured_at_utc=captured_at_utc,
        calibrator_loader=calibrator_loader,
        geometry_resolver=geometry_resolver,
        parse_fn=parse_fn,
    )
    toml_text = render_predictor_predictions_toml(
        captured_at_utc=captured_at_utc, rows=rows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml_text, encoding="utf-8")
    return output
