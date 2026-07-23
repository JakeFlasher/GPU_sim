"""Pooled cross-arch calibration head — feature-matrix builder
boundary.

This module owns the boundary between
:class:`baselines_harness.metrics.hardware_descriptor.HardwareDescriptor`
inputs and the numeric design matrix that the future trained head
consumes. The boundary lives here so any future training entry-point
(`fit_pooled_head`) calls into it and cannot extract features ad
hoc.

The builder is intentionally narrow:

* Accept ONLY a sequence of ``HardwareDescriptor`` instances.
* Reject dicts, strings, namedtuples / dataclasses of a different
  shape, and anything that exposes an arch-identifier field —
  these surfaces are how an opaque arch one-hot could otherwise
  enter training.
* Emit ``(POOLED_HEAD_FEATURE_NAMES, matrix)`` where ``matrix`` is
  a ``list[list[int]]`` of shape ``(N, 6)`` with values in
  canonical descriptor-field order.

No fitting logic, no model, no calibration semantics live here.
Those remain deferred until cross-arch label data is available.
"""

from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Final, Sequence

from baselines_harness.metrics.hardware_descriptor import (
    HardwareDescriptor,
    descriptor_field_names,
)


# The canonical feature-name tuple. Sourced from the descriptor
# field names at import time so the two sources cannot drift: if a
# future round adds a field to ``HardwareDescriptor``, this tuple
# updates automatically and the builder's row-construction loop
# uses the new field too.
POOLED_HEAD_FEATURE_NAMES: Final[tuple[str, ...]] = (
    descriptor_field_names()
)


POOLED_HEAD_FEATURE_COUNT: Final[int] = (
    len(POOLED_HEAD_FEATURE_NAMES)
)


# Canonical typed-reason vocabulary for builder fail-closed paths.
# Downstream callers compose typed-unavailable bundles from this
# vocabulary; new failure modes must be added here AND in the
# corresponding test_pooled_head_feature_matrix_error_typed_reasons_closed
# pin so the vocabulary stays explicit.
POOLED_HEAD_FEATURE_MATRIX_TYPED_REASONS: Final[frozenset[str]] = (
    frozenset({
        "pooled_head_feature_matrix_input_not_a_sequence",
        "pooled_head_feature_matrix_element_not_a_hardware_descriptor",
        "pooled_head_feature_matrix_empty_input",
        "pooled_head_feature_matrix_field_wrong_type",
        "pooled_head_feature_matrix_field_non_positive",
    })
)


class PooledHeadFeatureMatrixError(Exception):
    """Raised when
    :func:`build_pooled_head_feature_matrix` cannot build a
    six-column numeric matrix from the supplied input. Carries the
    canonical typed-reason string for downstream typed-unavailable
    bundle composition.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


def _require_plain_positive_int_field(
    descriptor: HardwareDescriptor,
    field_name: str,
    row_index: int,
) -> int:
    """Extract one field value from a ``HardwareDescriptor`` and
    runtime-validate it as a plain positive ``int``.

    ``HardwareDescriptor`` is a plain dataclass whose field
    annotations Python does not enforce at runtime. A directly
    constructed descriptor can therefore carry a string, a
    ``bool``, a float, or a non-positive integer in any of its
    six fields. The :func:`build_pooled_head_feature_matrix`
    builder calls into this extractor for every field so the
    matrix it emits is guaranteed numeric — values that pass
    Python's nominal type system but violate the contract are
    rejected before they enter the design matrix.

    The validation rules:

    * ``bool`` is rejected explicitly. ``True`` and ``False``
      satisfy ``isinstance(value, int)`` (because ``bool`` is an
      ``int`` subclass in Python), but they are semantically the
      wrong type for a hardware-feature column. Accepting them
      would let a directly-constructed descriptor with
      ``l1_kb_per_sm=True`` silently coerce to the integer ``1``.
    * Any non-``int`` value is rejected. Strings (including
      string-typed arch identifiers like ``"sm_86"``), floats,
      ``None``, and arbitrary objects all fail closed here.
    * Zero and negative values are rejected. Hardware features
      this builder consumes (cache capacities, line/sector sizes,
      slice counts, associativity) are all strictly positive in
      every real device card. A zero or negative value indicates
      a programming error, not a missing feature.

    Returns the validated ``int`` value on the happy path. Raises
    :class:`PooledHeadFeatureMatrixError` with one of:

    * ``pooled_head_feature_matrix_field_wrong_type`` — value is
      ``bool`` or not an ``int``.
    * ``pooled_head_feature_matrix_field_non_positive`` — value
      is zero or negative.
    """
    value = getattr(descriptor, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PooledHeadFeatureMatrixError(
            "pooled_head_feature_matrix_field_wrong_type",
            (
                f"row {row_index} descriptor field {field_name!r}"
                f"={value!r} (type {type(value).__name__!r}) is "
                f"not a plain int; the pooled-head feature matrix "
                f"requires numeric values only (bool subclass not "
                f"accepted)"
            ),
        )
    if value <= 0:
        raise PooledHeadFeatureMatrixError(
            "pooled_head_feature_matrix_field_non_positive",
            (
                f"row {row_index} descriptor field {field_name!r}"
                f"={value!r} is not positive; hardware-feature "
                f"columns require strictly positive integers"
            ),
        )
    return value


def build_pooled_head_feature_matrix(
    descriptors: Sequence[HardwareDescriptor],
) -> tuple[tuple[str, ...], list[list[int]]]:
    """Build the canonical six-column numeric feature matrix from a
    :class:`collections.abc.Sequence` of :class:`HardwareDescriptor`
    instances.

    Returns ``(POOLED_HEAD_FEATURE_NAMES, matrix)`` where ``matrix``
    is a ``list[list[int]]`` with one row per descriptor, six
    columns in canonical descriptor-field order.

    Accepted input shapes:

    * ``list[HardwareDescriptor]``
    * ``tuple[HardwareDescriptor, ...]``
    * any other ``collections.abc.Sequence`` subtype (positional
      indexing semantics) carrying ``HardwareDescriptor`` elements.

    Rejected input shapes:

    * ``str`` and ``bytes`` — both satisfy ``Sequence`` at the
      protocol level but carry character / byte elements rather
      than descriptor elements; accepting them would obscure the
      input contract. Routed to
      ``pooled_head_feature_matrix_input_not_a_sequence``.
    * Generators / iterators (``__iter__`` only, no
      ``__getitem__`` / ``__len__``) — single-pass iteration is
      a different protocol than ``Sequence``. Routed to
      ``pooled_head_feature_matrix_input_not_a_sequence``.
    * Bare ``dict`` / ``set`` / ``frozenset`` — neither is a
      ``Sequence`` (no positional indexing). Routed to
      ``pooled_head_feature_matrix_input_not_a_sequence``. The
      ``Sequence`` check is intentionally stricter than a bare
      iterability check so a bare dict (which would otherwise
      iterate over its keys) surfaces as an input-shape failure
      rather than as an element-type failure on the first key.
    * Any non-iterable input (int, None, opaque object). Routed
      to ``pooled_head_feature_matrix_input_not_a_sequence``.

    Per-element rejections (after the input-shape check passes):

    * ``pooled_head_feature_matrix_element_not_a_hardware_descriptor``
      — at least one Sequence element is not a
      ``HardwareDescriptor`` instance (lists of dicts with
      ``arch`` keys, lists of strings, lists of namedtuples /
      dataclasses of a different shape).
    * ``pooled_head_feature_matrix_field_wrong_type`` — a
      descriptor's field carries ``bool`` or a non-``int`` value
      at runtime. Defends against direct-construct bypass of
      dataclass annotations.
    * ``pooled_head_feature_matrix_field_non_positive`` — a
      descriptor's field carries zero or a negative value.
    * ``pooled_head_feature_matrix_empty_input`` — input is an
      empty Sequence; training requires at least one observation.
    """
    if isinstance(descriptors, (str, bytes)):
        raise PooledHeadFeatureMatrixError(
            "pooled_head_feature_matrix_input_not_a_sequence",
            (
                f"input is {type(descriptors).__name__!r}, not a "
                f"Sequence[HardwareDescriptor]"
            ),
        )
    if not isinstance(descriptors, collections.abc.Sequence):
        raise PooledHeadFeatureMatrixError(
            "pooled_head_feature_matrix_input_not_a_sequence",
            (
                f"input is {type(descriptors).__name__!r}, not a "
                f"collections.abc.Sequence; generators, dicts, "
                f"sets, and other iterables without positional "
                f"indexing are not accepted"
            ),
        )
    rows: list[list[int]] = []
    for idx, element in enumerate(descriptors):
        if not isinstance(element, HardwareDescriptor):
            raise PooledHeadFeatureMatrixError(
                "pooled_head_feature_matrix_element_not_a_hardware_descriptor",
                (
                    f"element at index {idx} is "
                    f"{type(element).__name__!r}, not a "
                    f"HardwareDescriptor instance"
                ),
            )
        rows.append([
            _require_plain_positive_int_field(element, name, idx)
            for name in POOLED_HEAD_FEATURE_NAMES
        ])
    if not rows:
        raise PooledHeadFeatureMatrixError(
            "pooled_head_feature_matrix_empty_input",
            "input sequence is empty; training requires at least "
            "one HardwareDescriptor observation",
        )
    return POOLED_HEAD_FEATURE_NAMES, rows


# --------------------------------------------------------------------
# Training entry-point boundary
# --------------------------------------------------------------------
#
# The pooled cross-arch head's training entry-point lives here so any
# future fitting implementation flows through the same descriptor
# validation + feature-matrix builder boundary the input contract
# requires. Today the training algorithm itself is deferred until
# cross-arch labels are harvested on real hardware; the entry-point
# below validates inputs and then raises a typed
# ``PooledHeadTrainingNotAvailableError`` on the post-validation
# training path. When labels arrive, someone replaces the raise with
# the real fitting code; the signature, validation chain, builder-
# reference body-scoped guard, and return-type contract are already
# locked.


POOLED_HEAD_TRAINING_TYPED_REASONS: Final[frozenset[str]] = frozenset({
    "pooled_head_training_labels_not_yet_harvested",
    "pooled_head_training_labels_count_mismatch",
    "pooled_head_training_labels_wrong_type",
    "pooled_head_training_labels_out_of_range",
    "pooled_head_fit_result_calibration_source_invalid",
    "pooled_head_fit_result_prediction_mode_invalid",
    "pooled_head_fit_result_confidence_band_invalid",
    "pooled_head_fit_result_b200_validated_wrong_type",
    "pooled_head_fit_result_zero_shot_provenance_invalid",
})


POOLED_HEAD_CALIBRATION_SOURCES: Final[frozenset[str]] = frozenset({
    "sm_86",
    "sm_120",
    "sm_86+sm_120",
})


# The pooled-head canonical prediction-mode vocabulary MUST match
# the deployed cache_prediction surface vocabulary
# (``CACHE_PREDICTION_PREDICTION_MODES`` in
# ``baselines_harness.surfaces.cache_prediction``). Any divergence
# would silently mis-route a ``PooledHeadFitResult`` into a surface
# prediction-mode rejection. A dedicated drift-guard test in
# ``test_pooled_cross_arch_head`` asserts the two frozensets are
# equal so future edits to either surface fail closed at test time.
POOLED_HEAD_PREDICTION_MODES: Final[frozenset[str]] = frozenset({
    "in_distribution",
    "zero_shot",
})


class PooledHeadTrainingNotAvailableError(Exception):
    """Raised by :func:`fit_pooled_head` or
    :class:`PooledHeadFitResult.__post_init__` when the training
    path / construction path cannot complete. Carries the canonical
    typed-reason string for downstream typed-unavailable bundle
    composition.

    The vocabulary covers nine distinct failure modes. The first
    four are :func:`fit_pooled_head` label-validation paths; the
    remaining five are :class:`PooledHeadFitResult` constructor
    validators (four per-field, one cross-field):

    * ``pooled_head_training_labels_not_yet_harvested`` — the
      requested fit is well-formed but no cross-arch label data is
      available in this environment. This is the expected raise
      when the function is called on the local sm_86 host before
      sm_120 hardware has been provisioned.
    * ``pooled_head_training_labels_count_mismatch`` — the
      ``descriptors`` and ``labels`` sequences have different
      lengths.
    * ``pooled_head_training_labels_wrong_type`` — covers two
      container-shape and per-element type failure modes: (a) the
      ``labels`` argument is not a
      ``collections.abc.Sequence`` (rejects str/bytes containers,
      generators, dicts, sets, None, ints, opaque objects); (b)
      at least one label element is bool or not a plain
      ``float``/``int``.
    * ``pooled_head_training_labels_out_of_range`` — at least one
      label is a finite numeric type but its value lies outside
      the NCU hit-rate domain ``[0.0, 1.0]`` (rejects ``nan``,
      ``inf``, negative values, and values above 1).
    * ``pooled_head_fit_result_calibration_source_invalid`` —
      ``calibration_source`` is not in
      :data:`POOLED_HEAD_CALIBRATION_SOURCES`.
    * ``pooled_head_fit_result_prediction_mode_invalid`` —
      ``prediction_mode`` is not in
      :data:`POOLED_HEAD_PREDICTION_MODES`.
    * ``pooled_head_fit_result_confidence_band_invalid`` —
      ``confidence_band`` is not ``None`` and is not a 2-tuple of
      finite floats in ``[0.0, 1.0]`` with ``lower <= upper``.
    * ``pooled_head_fit_result_b200_validated_wrong_type`` —
      ``b200_validated`` is not exactly a ``bool`` (int subclass
      coercion rejected via ``type() is bool``).
    * ``pooled_head_fit_result_zero_shot_provenance_invalid`` —
      ``prediction_mode == "zero_shot"`` is combined with a
      provenance shape that contradicts the deployed B200 zero-
      shot contract (cross-arch pooled source, populated
      confidence band, pre-online-judge B200 flag). The cross-
      field validator fires only if all four per-field validators
      have already accepted the inputs.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class PooledHeadFitResult:
    """Return type for :func:`fit_pooled_head`. Frozen so consumers
    can cache it per training run.

    The four fields lock the deployment-provenance contract the
    eventual B200 zero-shot cache-prediction surface must emit:

    * ``calibration_source`` — one of
      :data:`POOLED_HEAD_CALIBRATION_SOURCES` (`"sm_86"`,
      `"sm_120"`, `"sm_86+sm_120"`). Identifies which arch's
      labels trained the head.
    * ``prediction_mode`` — one of
      :data:`POOLED_HEAD_PREDICTION_MODES`
      (`"in_distribution"`, `"zero_shot"`). Defaults to
      `"in_distribution"` (the on-arch calibrated mode used when
      the labels came from the same arch the predictor will run
      on); the B200 deployment path emits `"zero_shot"`. The
      vocabulary mirrors
      :data:`baselines_harness.surfaces.cache_prediction.CACHE_PREDICTION_PREDICTION_MODES`
      so any future ``PooledHeadFitResult`` round-trips into the
      deployed surface without a vocabulary-rejection.
    * ``confidence_band`` — pair of finite floats
      `(lower_residual, upper_residual)` in `[0.0, 1.0]` with
      `lower <= upper`, or `None` when no band has been
      computed yet.
    * ``b200_validated`` — explicit boolean (not
      ``int``-subclass). Defaults to ``False`` pre-online-judge
      submission.

    Constructor-time validation lives in :meth:`__post_init__`;
    every adversarial-input path fail-closes via
    :class:`PooledHeadTrainingNotAvailableError` with a canonical
    typed reason in
    :data:`POOLED_HEAD_TRAINING_TYPED_REASONS`. The four per-field
    validators run first; the cross-field zero-shot validator
    runs last and fires only when ``prediction_mode == "zero_shot"``
    is combined with a provenance shape that contradicts the
    deployed B200 zero-shot contract (cross-arch pooled source,
    populated confidence band, pre-online-judge B200 flag).
    """

    calibration_source: str
    prediction_mode: str = "in_distribution"
    confidence_band: tuple[float, float] | None = None
    b200_validated: bool = False

    def __post_init__(self) -> None:
        _validate_calibration_source(self.calibration_source)
        _validate_prediction_mode(self.prediction_mode)
        _validate_confidence_band(self.confidence_band)
        _validate_b200_validated(self.b200_validated)
        # Cross-field validator runs LAST so it can assume every
        # per-field invariant already held. Splitting it out keeps
        # each per-field validator focused on its own field; this
        # one focuses on the inter-field provenance shape the B200
        # zero-shot emission contract requires.
        self._validate_zero_shot_cross_field_invariant()

    def _validate_zero_shot_cross_field_invariant(self) -> None:
        """Reject any ``prediction_mode == "zero_shot"``
        construction whose provenance shape contradicts the
        deployed B200 zero-shot cache-prediction surface emission
        contract.

        The zero-shot contract requires all three of:

        * ``calibration_source == "sm_86+sm_120"`` — zero-shot is
          a cross-arch prediction; an on-arch source string is
          incompatible with the mode label.
        * ``confidence_band is not None`` — a deployed zero-shot
          emission must carry the LOOCV residual descriptor; a
          ``None`` band represents "not yet computed" which is
          incompatible with deployment.
        * ``b200_validated is False`` — the flag flips to ``True``
          only after the deployed-side hook has run and emitted a
          confirmed B200 artifact. Constructing a result with
          ``b200_validated=True`` from this code path is a
          provenance lie; the deployed hook is the only writer.

        Any violation raises
        :class:`PooledHeadTrainingNotAvailableError` with typed
        reason
        ``pooled_head_fit_result_zero_shot_provenance_invalid``
        and a detail message naming the offending field. The
        validator is a no-op for non-zero-shot modes.
        """
        if self.prediction_mode != "zero_shot":
            return
        if self.calibration_source != "sm_86+sm_120":
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_fit_result_zero_shot_provenance_invalid",
                (
                    f"prediction_mode='zero_shot' requires "
                    f"calibration_source='sm_86+sm_120' (the "
                    f"cross-arch pooled source); got "
                    f"calibration_source={self.calibration_source!r}"
                ),
            )
        if self.confidence_band is None:
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_fit_result_zero_shot_provenance_invalid",
                (
                    "prediction_mode='zero_shot' requires a "
                    "populated confidence_band (the LOOCV "
                    "residual descriptor); got confidence_band=None"
                ),
            )
        if self.b200_validated is not False:
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_fit_result_zero_shot_provenance_invalid",
                (
                    f"prediction_mode='zero_shot' requires "
                    f"b200_validated=False until the deployed-side "
                    f"hook has run and emitted a confirmed B200 "
                    f"artifact; got "
                    f"b200_validated={self.b200_validated!r}"
                ),
            )


def _validate_calibration_source(value: str) -> None:
    """Reject any ``calibration_source`` string that is not in
    :data:`POOLED_HEAD_CALIBRATION_SOURCES`. Future calibration
    runs that add a new arch must update the frozenset AND the
    closed-vocabulary test in lockstep.
    """
    if not isinstance(value, str) or value not in POOLED_HEAD_CALIBRATION_SOURCES:
        raise PooledHeadTrainingNotAvailableError(
            "pooled_head_fit_result_calibration_source_invalid",
            (
                f"calibration_source={value!r} (type "
                f"{type(value).__name__!r}) is not in the "
                f"canonical set "
                f"{sorted(POOLED_HEAD_CALIBRATION_SOURCES)!r}"
            ),
        )


def _validate_prediction_mode(value: str) -> None:
    """Reject any ``prediction_mode`` string that is not in
    :data:`POOLED_HEAD_PREDICTION_MODES`.
    """
    if not isinstance(value, str) or value not in POOLED_HEAD_PREDICTION_MODES:
        raise PooledHeadTrainingNotAvailableError(
            "pooled_head_fit_result_prediction_mode_invalid",
            (
                f"prediction_mode={value!r} (type "
                f"{type(value).__name__!r}) is not in the "
                f"canonical set "
                f"{sorted(POOLED_HEAD_PREDICTION_MODES)!r}"
            ),
        )


def _validate_confidence_band(
    value: tuple[float, float] | None,
) -> None:
    """Reject any ``confidence_band`` that is not ``None`` and is
    not a 2-tuple of finite floats in ``[0.0, 1.0]`` with
    ``lower <= upper``.
    """
    if value is None:
        return
    if not isinstance(value, tuple) or len(value) != 2:
        raise PooledHeadTrainingNotAvailableError(
            "pooled_head_fit_result_confidence_band_invalid",
            (
                f"confidence_band={value!r} (type "
                f"{type(value).__name__!r}) is not a 2-tuple; "
                f"must be (lower_residual, upper_residual) or "
                f"None"
            ),
        )
    lower, upper = value
    for name, scalar in (("lower", lower), ("upper", upper)):
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_fit_result_confidence_band_invalid",
                (
                    f"confidence_band {name}={scalar!r} (type "
                    f"{type(scalar).__name__!r}) is not a plain "
                    f"int or float"
                ),
            )
        scalar_float = float(scalar)
        if math.isnan(scalar_float) or math.isinf(scalar_float):
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_fit_result_confidence_band_invalid",
                (
                    f"confidence_band {name}={scalar!r} is not "
                    f"finite; bounds must be finite floats in "
                    f"[0.0, 1.0]"
                ),
            )
        if scalar_float < 0.0 or scalar_float > 1.0:
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_fit_result_confidence_band_invalid",
                (
                    f"confidence_band {name}={scalar!r} lies "
                    f"outside [0.0, 1.0]"
                ),
            )
    if float(lower) > float(upper):
        raise PooledHeadTrainingNotAvailableError(
            "pooled_head_fit_result_confidence_band_invalid",
            (
                f"confidence_band lower={lower!r} > "
                f"upper={upper!r}; lower bound must not exceed "
                f"upper bound"
            ),
        )


def _validate_b200_validated(value: bool) -> None:
    """Reject any ``b200_validated`` that is not exactly a
    ``bool``. ``int`` subclasses (including ``True``/``False``
    via their ``int``-conformance) are normally interchangeable
    with bool, but the deployment provenance flag is
    type-discriminated to prevent silent coercion of
    ``b200_validated=1`` into ``True``.
    """
    # ``type(value) is bool`` (vs ``isinstance(value, bool)``)
    # rejects any future bool subclass. ``bool`` IS an ``int``
    # subclass, so ``isinstance(1, bool)`` is ``False`` but
    # ``isinstance(True, int)`` is ``True`` — the ``type() is``
    # check is the strictest form available.
    if type(value) is not bool:
        raise PooledHeadTrainingNotAvailableError(
            "pooled_head_fit_result_b200_validated_wrong_type",
            (
                f"b200_validated={value!r} (type "
                f"{type(value).__name__!r}) is not a plain "
                f"bool; int subclass not accepted"
            ),
        )


def _require_valid_label_sequence(
    descriptors: Sequence[HardwareDescriptor],
    labels: Sequence[float],
) -> None:
    """Validate the ``labels`` sequence against the
    ``descriptors`` sequence.

    Enforcement order (fail-close at first violation):

    1. ``labels`` is not ``str`` / ``bytes`` (both are
       ``Sequence``-shaped at the protocol level but carry
       character/byte elements that would otherwise iterate as
       individual labels).
    2. ``labels`` is a ``collections.abc.Sequence`` (rejects
       generators, dicts, sets, ``None``, ints, and opaque
       objects with the same precision the descriptor builder
       uses).
    3. ``len(labels) == len(descriptors)``.
    4. For each label element: not ``bool``, is ``int`` or
       ``float`` (numeric type check).
    5. For each label element: finite (rejects ``nan`` and
       ``inf`` via :func:`math.isnan` / :func:`math.isinf`).
    6. For each label element: ``0.0 <= label <= 1.0`` (NCU
       hit-rate domain).

    Raises :class:`PooledHeadTrainingNotAvailableError`:

    * ``pooled_head_training_labels_wrong_type`` — container is
      not a Sequence (or is str/bytes), or a label element fails
      the numeric type check.
    * ``pooled_head_training_labels_count_mismatch`` — count
      mismatch with ``descriptors``.
    * ``pooled_head_training_labels_out_of_range`` — a label is
      non-finite or outside ``[0.0, 1.0]``.
    """
    if isinstance(labels, (str, bytes)):
        raise PooledHeadTrainingNotAvailableError(
            "pooled_head_training_labels_wrong_type",
            (
                f"labels container is {type(labels).__name__!r}, "
                f"not a Sequence[float]; str/bytes containers "
                f"would otherwise iterate as individual labels"
            ),
        )
    if not isinstance(labels, collections.abc.Sequence):
        raise PooledHeadTrainingNotAvailableError(
            "pooled_head_training_labels_wrong_type",
            (
                f"labels container is {type(labels).__name__!r}, "
                f"not a collections.abc.Sequence; generators, "
                f"dicts, sets, None, ints, and other "
                f"non-Sequence inputs are not accepted"
            ),
        )
    if len(labels) != len(descriptors):
        raise PooledHeadTrainingNotAvailableError(
            "pooled_head_training_labels_count_mismatch",
            (
                f"got {len(labels)} labels for {len(descriptors)} "
                f"descriptors; counts must match"
            ),
        )
    for idx, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, (int, float)):
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_training_labels_wrong_type",
                (
                    f"label at index {idx}={label!r} (type "
                    f"{type(label).__name__!r}) is not a plain "
                    f"int or float; bool subclass not accepted"
                ),
            )
        label_float = float(label)
        if math.isnan(label_float) or math.isinf(label_float):
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_training_labels_out_of_range",
                (
                    f"label at index {idx}={label!r} is not "
                    f"finite; NCU hit-rate labels must be finite "
                    f"fractions in [0.0, 1.0]"
                ),
            )
        if label_float < 0.0 or label_float > 1.0:
            raise PooledHeadTrainingNotAvailableError(
                "pooled_head_training_labels_out_of_range",
                (
                    f"label at index {idx}={label!r} lies "
                    f"outside the NCU hit-rate domain [0.0, 1.0]"
                ),
            )


def fit_pooled_head(
    descriptors: Sequence[HardwareDescriptor],
    labels: Sequence[float],
) -> PooledHeadFitResult:
    """Train the pooled cross-arch calibration head over a set of
    :class:`HardwareDescriptor` observations and their NCU-measured
    cache-prediction labels.

    The function first validates the descriptor sequence via
    :func:`build_pooled_head_feature_matrix` (so all six-column
    matrix contract assertions fire before any fitting logic
    runs), then validates the label sequence against the
    descriptors, then attempts to fit.

    Today the actual fitting algorithm is deferred: cross-arch
    label data is harvested from an NCU calibration sweep on
    sm_120 hardware that is not yet available in this environment.
    The function raises
    :class:`PooledHeadTrainingNotAvailableError` with typed reason
    ``pooled_head_training_labels_not_yet_harvested`` on the
    post-validation path. The signature, validation chain,
    builder reference, and return-type are all locked here so the
    future fitting implementation cannot drift away from the
    descriptor-only numeric-feature contract.

    Per-call failure modes (in evaluation order):

    1. :class:`PooledHeadFeatureMatrixError` — propagated from
       :func:`build_pooled_head_feature_matrix` when the
       descriptor sequence violates the input contract.
    2. ``PooledHeadTrainingNotAvailableError(pooled_head_training_labels_wrong_type)``
       — the ``labels`` container is ``str`` / ``bytes`` /
       non-``Sequence``, or at least one label element is
       ``bool`` / non-``int``/``float``.
    3. ``PooledHeadTrainingNotAvailableError(pooled_head_training_labels_count_mismatch)``
       — descriptor and label counts disagree (after the
       container-shape check passes).
    4. ``PooledHeadTrainingNotAvailableError(pooled_head_training_labels_out_of_range)``
       — at least one label is non-finite (``nan`` / ``inf``)
       or lies outside the NCU hit-rate domain ``[0.0, 1.0]``.
    5. ``PooledHeadTrainingNotAvailableError(pooled_head_training_labels_not_yet_harvested)``
       — all inputs are valid; sm_120 hardware is not provisioned
       so no fit can run yet.
    """
    # Reference the builder explicitly so the body-scoped
    # negative-regression guard in
    # ``test_pooled_head_training_entry_references_builder_boundary``
    # passes via ``inspect.getsource`` scan.
    _names, _matrix = build_pooled_head_feature_matrix(descriptors)
    _require_valid_label_sequence(descriptors, labels)
    raise PooledHeadTrainingNotAvailableError(
        "pooled_head_training_labels_not_yet_harvested",
        (
            "pooled cross-arch head training requires sm_120 NCU "
            "calibration labels that are not yet available in "
            "this environment; the descriptor + label inputs "
            "validated cleanly but no fit can run until sm_120 "
            "hardware is provisioned"
        ),
    )


def make_b200_zero_shot_fit_result(
    confidence_band: tuple[float, float],
) -> PooledHeadFitResult:
    """Construct the canonical B200 zero-shot
    :class:`PooledHeadFitResult` shape from a single LOOCV
    residual band tuple.

    The eventual B200 deployment hook constructs a
    ``PooledHeadFitResult`` with the four fields locked by the
    deployment-provenance contract:

    * ``calibration_source="sm_86+sm_120"`` — the cross-arch
      pooled source the zero-shot mode requires.
    * ``prediction_mode="zero_shot"`` — the B200 deployment
      mode (no on-arch calibration sweep was run).
    * ``confidence_band=<LOOCV residual pair>`` — the only
      field that varies per training run.
    * ``b200_validated=False`` — the flag flips to ``True``
      only via the deployed-side hook AFTER the actual
      artifact emission, not at construction time. The
      deployed-side flip is a separate code path that does
      NOT route through this factory.

    Three of the four fields are constants for every B200
    zero-shot construction. This factory hardcodes them in
    code so call sites pass only the variable band, removing
    the chance of a typo across the canonical-value fields
    and reducing the eventual training code's surface area.

    The factory delegates input validation to
    :meth:`PooledHeadFitResult.__post_init__`, which runs the
    per-field validators FIRST and the cross-field zero-shot
    invariant LAST. The two failure paths surface different
    canonical typed reasons:

    * Per-field band failures — non-tuple input, wrong shape
      (not exactly two elements), non-finite endpoint
      (``nan`` / ``inf``), endpoint outside ``[0.0, 1.0]``, or
      inverted ``lower > upper`` — raise
      :class:`PooledHeadTrainingNotAvailableError` with the
      existing typed reason
      ``pooled_head_fit_result_confidence_band_invalid``.
    * Cross-field zero-shot invariant failure — since this
      factory always sets ``prediction_mode="zero_shot"``,
      ``calibration_source="sm_86+sm_120"``, and
      ``b200_validated=False``, the only cross-field
      violation reachable through this entry point is the
      ``confidence_band is None`` case. The per-field
      ``_validate_confidence_band`` validator accepts
      ``None`` as the "not yet computed" sentinel (in-
      distribution construction may leave the band
      unpopulated), so ``None`` passes through per-field
      validation and is caught by the cross-field invariant.
      The cross-field validator raises with the existing
      typed reason
      ``pooled_head_fit_result_zero_shot_provenance_invalid``
      because a deployed B200 zero-shot emission requires a
      populated LOOCV residual band.

    No new typed reasons are introduced; the factory leverages
    the existing 9-element
    :data:`POOLED_HEAD_TRAINING_TYPED_REASONS` frozenset.

    Downstream callers chain the factory through
    :func:`baselines_harness.surfaces.cache_prediction.cache_prediction_provenance_from_pooled_head_fit_result`
    to obtain the surface-side
    :class:`CachePredictionProvenance` for manifest emission.
    """
    return PooledHeadFitResult(
        calibration_source="sm_86+sm_120",
        prediction_mode="zero_shot",
        confidence_band=confidence_band,
        b200_validated=False,
    )


__all__ = [
    "POOLED_HEAD_FEATURE_NAMES",
    "POOLED_HEAD_FEATURE_COUNT",
    "POOLED_HEAD_FEATURE_MATRIX_TYPED_REASONS",
    "POOLED_HEAD_TRAINING_TYPED_REASONS",
    "POOLED_HEAD_CALIBRATION_SOURCES",
    "POOLED_HEAD_PREDICTION_MODES",
    "PooledHeadFeatureMatrixError",
    "PooledHeadTrainingNotAvailableError",
    "PooledHeadFitResult",
    "build_pooled_head_feature_matrix",
    "fit_pooled_head",
    "make_b200_zero_shot_fit_result",
]
