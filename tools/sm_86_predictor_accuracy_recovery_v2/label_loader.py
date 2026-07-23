"""Mode-aware label loader for the submission-track
operational blinding contract.

The cohort split locks the 10 holdout cells' truth values into
``data/sm_86_predictor_accuracy_recovery_v2/sealed_holdout_labels.json``.
The calibration runner and the validation runner MUST NOT open
that file: the submission-track contract treats it as the
load-bearing defense against calibration leakage. Only the
holdout-evaluation runner (after a passing validation pass) is
permitted to read those rows, and only while the
``unblind_token`` ContextVar holds ``True``.

This module deliberately does NOT open the sealed labels file
from any helper that runs in calibration/validation mode. The
holdout-key set used by filter / assert helpers is derived
from ``cohort_split.json`` (a non-truth metadata artifact
carrying identity tuples only), so calibration-side code paths
never touch the sealed truths.

Exports:
  - ``unblind_token`` ContextVar (default ``False``): the gate
    holdout-side code flips with :func:`holdout_unblind_context`
    when reading sealed labels.
  - :func:`load_labels`: mode-aware loader. If the requested
    path resolves to the sealed file AND ``unblind_token`` is
    not set, raises :class:`BlindedSealedLabelAccessError`.
    Otherwise loads normally.
  - :func:`holdout_unblind_context`: contextmanager that flips
    the ContextVar to ``True`` for the lifetime of the
    ``with`` block.
  - :func:`assert_path_is_not_sealed`: operational
    path-refusal helper. Calibration / validation runner
    entrypoints call this BEFORE any label I/O so the runner
    fails closed when the operator passes the sealed path.
  - :func:`holdout_keys_from_cohort_split`: derives the set of
    `(slug, shape_point)` keys from the live ``cohort_split.json``
    holdout block (or a caller-supplied split path). Returns
    identity keys only -- no truth values.
  - :func:`assert_payload_has_no_holdout_truth_rows`: typed
    raise when an already-built label payload (e.g., a cached
    preload) contains any holdout `(slug, shape_point)` key
    while the caller is in calibration / validation mode.
  - :func:`filter_admitted_labels_to_non_sealed_keys`: helper
    that returns the subset of an admitted-label corpus payload
    excluding holdout identity keys. Used to construct safe
    calibration/validation label maps.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping


_REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared mixed-arity corpus-cell identity helper. For a base cell
# corpus_cell_key yields (slug, shape_point) — byte-identical to the historical
# holdout key, so threading it is a no-op on the current all-base cohort. Once a
# register-cap build variant enters a split it widens to
# (slug, shape_point, variant_id), so holdout-blinding filters/asserts the
# SPECIFIC variant rather than collapsing every sibling of one (slug, shape).
_LOCAL_LABELS_DIR = _REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS_DIR))
from corpus_cell_identity import corpus_cell_key  # noqa: E402
SEALED_HOLDOUT_LABELS_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "sealed_holdout_labels.json"
)
COHORT_SPLIT_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_split.json"
)


# Default to False: the loader refuses to open the sealed file
# unless an explicit holdout-mode context manager flips this to
# True.
unblind_token: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unblind_token", default=False,
)


class BlindedSealedLabelAccessError(Exception):
    """Raised when calibration- or validation-mode code attempts
    to read holdout truth values (either by opening the sealed
    file directly or by accessing a cached preload that
    includes holdout rows) without the ``unblind_token``
    ContextVar set true. This exception is the load-bearing
    operational-blinding surface."""


def _resolves_to_sealed(path: Path) -> bool:
    """Return True iff ``path`` resolves to the sealed file.
    Comparison is by resolved absolute path so symlinks /
    ``..`` / relative paths cannot bypass the check."""
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    try:
        sealed_resolved = SEALED_HOLDOUT_LABELS_PATH.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return resolved == sealed_resolved


def assert_path_is_not_sealed(path: Path | str, mode: str) -> None:
    """Operational path-refusal rule for calibration + validation
    runner entrypoints. If ``mode`` is ``"calibration"`` or
    ``"validation"`` AND ``path`` resolves to the sealed file,
    raise :class:`BlindedSealedLabelAccessError` BEFORE any I/O.

    Holdout mode lets the path through; the ContextVar guard
    inside :func:`load_labels` is the second layer of defense
    for that path. Both layers must be in place for the
    submission-track blinding contract.
    """
    if mode not in ("calibration", "validation", "holdout"):
        raise ValueError(
            f"mode must be one of 'calibration' / 'validation' / "
            f"'holdout'; got {mode!r}"
        )
    path = Path(path)
    if mode in ("calibration", "validation") and _resolves_to_sealed(path):
        raise BlindedSealedLabelAccessError(
            f"operational refusal: {mode}-mode runner cannot "
            f"accept the sealed holdout labels path "
            f"({SEALED_HOLDOUT_LABELS_PATH}); only the holdout-"
            f"evaluation runner may consume this file, and only "
            f"after a passing validation pass with the "
            f"unblind_token ContextVar set true"
        )


def load_labels(path: Path | str) -> dict:
    """Load a JSON label corpus from ``path``. If ``path``
    resolves to the sealed file AND the ``unblind_token``
    ContextVar is not set true, raise
    :class:`BlindedSealedLabelAccessError` instead of opening
    the file. This is the code-level defense complementing the
    operational :func:`assert_path_is_not_sealed`."""
    path = Path(path)
    if _resolves_to_sealed(path) and not unblind_token.get():
        raise BlindedSealedLabelAccessError(
            f"blinded access: label loader refused to open "
            f"sealed file {path} because unblind_token is False. "
            f"Wrap the holdout-evaluation entrypoint in "
            f"holdout_unblind_context() to permit the read."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@contextlib.contextmanager
def holdout_unblind_context() -> Iterator[None]:
    """Context manager that flips ``unblind_token`` to ``True``
    for the duration of the ``with`` block, then restores the
    previous value. Holdout-evaluation code is expected to wrap
    its load with this context manager."""
    token = unblind_token.set(True)
    try:
        yield
    finally:
        unblind_token.reset(token)


def holdout_keys_from_cohort_split(
    cohort_split_path: Path | str | None = None,
) -> set[tuple[str, str]]:
    """Return the set of `(slug, shape_point)` keys present in
    the holdout block of the paired ``cohort_split.json``. The
    cohort split is a NON-TRUTH metadata artifact: it carries
    identity tuples but not labels. Reading it from
    calibration/validation code paths is safe because no truth
    values cross.

    Calibration- and validation-side helpers (such as
    :func:`filter_admitted_labels_to_non_sealed_keys` and
    :func:`assert_payload_has_no_holdout_truth_rows`) call this
    function to obtain the holdout-key set without ever opening
    the sealed labels file.
    """
    path = Path(cohort_split_path) if cohort_split_path else COHORT_SPLIT_PATH
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    holdout_cells = split.get("splits", {}).get("holdout", {}).get("cells", [])
    return {corpus_cell_key(c, with_arch=False) for c in holdout_cells}


def assert_payload_has_no_holdout_truth_rows(
    payload: Mapping[str, Any],
    mode: str,
    holdout_keys: set[tuple[str, str]] | None = None,
) -> None:
    """Typed-raise helper for cached-preload protection. If
    ``mode in {"calibration", "validation"}`` AND ``payload``'s
    ``cells`` list contains any cell whose
    ``(slug, shape_point)`` is in ``holdout_keys``, raise
    :class:`BlindedSealedLabelAccessError`. Caller may pass an
    explicit ``holdout_keys`` set; default is derived from
    :func:`holdout_keys_from_cohort_split`.

    This complements :func:`filter_admitted_labels_to_non_sealed_keys`:
    the filter returns a safe payload; this assertion catches
    a payload that was constructed WITHOUT going through the
    filter and still carries holdout rows. The plan's
    leak-path-(b) negative test asserts the typed raise."""
    if mode not in ("calibration", "validation", "holdout"):
        raise ValueError(
            f"mode must be one of 'calibration' / 'validation' / "
            f"'holdout'; got {mode!r}"
        )
    if mode == "holdout":
        return
    if holdout_keys is None:
        holdout_keys = holdout_keys_from_cohort_split()
    if not holdout_keys:
        return
    leaks = []
    for cell in payload.get("cells", []):
        key = corpus_cell_key(cell, with_arch=False)
        if key in holdout_keys:
            leaks.append(key)
    if leaks:
        raise BlindedSealedLabelAccessError(
            f"cached-preload leak: {mode}-mode payload includes "
            f"{len(leaks)} holdout cell(s) (e.g., {sorted(leaks)[:3]!r}); "
            f"route the payload through "
            f"filter_admitted_labels_to_non_sealed_keys() before "
            f"caching or move the holdout reads under "
            f"holdout_unblind_context()"
        )


def load_labels_for_mode(
    path: Path | str,
    mode: str,
    *,
    strict_no_holdout_rows: bool = False,
) -> dict:
    """Canonical mode-aware loader for every calibration /
    validation / holdout label-input entrypoint. Routes the
    operational-blinding contract through one helper so a
    reviewer can grep every conversion site by name.

    Flow:
      1. :func:`assert_path_is_not_sealed` fails closed if
         ``path`` resolves to the sealed file and ``mode`` is
         ``"calibration"`` or ``"validation"``. Returns
         without raising for ``"holdout"``.
      2. :func:`load_labels` opens the file. The ContextVar
         guard inside :func:`load_labels` is the second layer
         of defense: if ``mode == "holdout"`` AND the caller
         forgot to wrap with :func:`holdout_unblind_context`,
         the read still fails closed.
      3. If ``strict_no_holdout_rows=True`` AND ``mode`` is
         ``"calibration"`` or ``"validation"``,
         :func:`assert_payload_has_no_holdout_truth_rows`
         runs against the loaded payload. Strict mode is for
         callers that promise their cached maps will not
         include any holdout truth row; callers that
         legitimately consume the full admitted-label corpus
         (which DOES contain holdout rows by default) keep
         the default ``strict_no_holdout_rows=False``.

    Use this helper at every entrypoint that takes a
    user-supplied label path. The grep target
    ``label_loader.load_labels_for_mode`` is the single
    enumerable signature for the entire blinding surface.
    """
    if mode not in ("calibration", "validation", "holdout"):
        raise ValueError(
            f"mode must be one of 'calibration' / 'validation' / "
            f"'holdout'; got {mode!r}"
        )
    assert_path_is_not_sealed(path, mode)
    payload = load_labels(path)
    if strict_no_holdout_rows and mode in ("calibration", "validation"):
        assert_payload_has_no_holdout_truth_rows(payload, mode)
    return payload


def filter_admitted_labels_to_non_sealed_keys(
    admitted_payload: Mapping[str, Any],
    holdout_keys: set[tuple[str, str]] | None = None,
) -> dict:
    """Return a deep-copied admitted-label payload with rows
    whose ``(slug, shape_point)`` are in the holdout-key set
    REMOVED. The holdout-key set defaults to the result of
    :func:`holdout_keys_from_cohort_split`, so this helper
    never opens the sealed truths file.

    The output is a NEW dict; the input is not mutated. The
    output's ``cells`` list is sorted by ``(slug, shape_point)``
    for determinism.
    """
    if holdout_keys is None:
        holdout_keys = holdout_keys_from_cohort_split()
    cells = list(admitted_payload.get("cells", []))
    filtered = [
        c for c in cells
        if corpus_cell_key(c, with_arch=False) not in holdout_keys
    ]
    filtered.sort(
        key=lambda c: (
            c.get("slug", ""), c.get("shape_point", ""),
            c.get("variant_id") or "",
        )
    )
    out = dict(admitted_payload)
    out["cells"] = filtered
    return out
