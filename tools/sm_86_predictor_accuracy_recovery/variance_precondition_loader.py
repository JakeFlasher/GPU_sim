"""Variance-precondition artifact loader.

Reads the post-refinement variance trajectory artifact at
``data/sm_86_predictor_accuracy_recovery/ac6_variance_trajectory_post_ac4.json``
and returns the load-bearing `current_stddev` for downstream
producer callers (the fail-closed sweep regeneration script and
the production headline orchestrator).

The loader asserts the variance precondition's status matches the
expected value for the calling context (the fail-closed sweep
regeneration requires ``precondition_status == "not_satisfied"`` so
the generated bundles are coherent with the verdict; a future
apply-arm caller would assert ``"satisfied"``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANCE_ARTIFACT_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "ac6_variance_trajectory_post_ac4.json"
)


class VariancePrecondition(NamedTuple):
    """Loaded variance-precondition state.

    Attributes:
        current_stddev: the cross-cell prediction stddev computed
            from the 5-cell admitted activation cohort.
        documented_floor_stddev: the documented floor (typically
            0.10 per the loop's DEC-2 / DEC-4 adjudications).
        precondition_status: ``"satisfied"`` or ``"not_satisfied"``.
        artifact_path: the file the values were loaded from.
    """

    current_stddev: float
    documented_floor_stddev: float
    precondition_status: str
    artifact_path: Path


def load_variance_precondition(
    artifact_path: Path | None = None,
) -> VariancePrecondition:
    """Load the variance-precondition state from the committed
    artifact.

    Defaults to the committed loop artifact at
    ``DEFAULT_VARIANCE_ARTIFACT_PATH``. Returns a NamedTuple with
    the load-bearing fields. Does NOT assert any precondition_status
    value — callers do their own assertion based on their context.

    Raises ``FileNotFoundError`` when the artifact is missing,
    ``ValueError`` when a required field is missing or has the wrong
    type (the upstream artifact scanner already validates the shape,
    but this loader is defensive against partial / corrupted files).
    """
    if artifact_path is None:
        artifact_path = DEFAULT_VARIANCE_ARTIFACT_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"variance trajectory artifact not found: {artifact_path}"
        )
    with artifact_path.open("r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    # Field-name back-compat: the v1 trajectory artifact under
    # data/sm_86_predictor_accuracy_recovery/ encodes the stddev as
    # `current_stddev`; the v2 trajectory under
    # data/sm_86_predictor_accuracy_recovery_v2/ encodes it as
    # `cross_cell_stddev` (the field name was clarified during the
    # corpus-expansion sweep). Accept either field; prefer
    # `current_stddev` when both are present (v1 contract is the
    # historical authority).
    current_stddev = artifact.get("current_stddev")
    if current_stddev is None:
        current_stddev = artifact.get("cross_cell_stddev")
    if not isinstance(current_stddev, (int, float)):
        raise ValueError(
            "current_stddev / cross_cell_stddev must be numeric;"
            f" got {current_stddev!r}"
        )
    floor = artifact.get("documented_floor_stddev")
    if not isinstance(floor, (int, float)):
        raise ValueError(
            f"documented_floor_stddev must be numeric; got {floor!r}"
        )
    status = artifact.get("precondition_status")
    if status not in ("satisfied", "not_satisfied"):
        raise ValueError(
            "precondition_status must be 'satisfied' or "
            f"'not_satisfied'; got {status!r}"
        )
    return VariancePrecondition(
        current_stddev=float(current_stddev),
        documented_floor_stddev=float(floor),
        precondition_status=str(status),
        artifact_path=artifact_path,
    )


def load_for_fail_closed_emission(
    artifact_path: Path | None = None,
) -> VariancePrecondition:
    """Load the variance precondition for the fail-closed sweep
    regeneration path. Asserts ``precondition_status ==
    "not_satisfied"``: if the artifact ever flips to
    ``"satisfied"`` (a future cohort expansion satisfies the
    variance floor), this loader call MUST fail so the operator
    re-thinks before emitting stale fail-closed bundles.

    Raises ``ValueError`` when the precondition is satisfied (in
    which case the calling script should not emit fail-closed
    bundles).
    """
    precondition = load_variance_precondition(artifact_path)
    if precondition.precondition_status != "not_satisfied":
        raise ValueError(
            "fail-closed sweep emission requires "
            "precondition_status == 'not_satisfied'; the loaded "
            f"artifact at {precondition.artifact_path} has "
            f"precondition_status == {precondition.precondition_status!r}. "
            "If a future cohort expansion has flipped the verdict "
            "to satisfied, the fail-closed sweep is stale and "
            "should be retired or regenerated via a different path."
        )
    return precondition
