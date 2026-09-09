"""Numeric hardware descriptor for the pooled cross-arch
calibration head.

The pooled cross-arch calibration head consumes a fixed-shape
covariate vector of NUMERIC hardware features:

* ``sector_size_bytes``     — L1TEX sector size (bytes).
* ``line_size_bytes``       — Cache line size (bytes).
* ``l1_kb_per_sm``          — L1 capacity per SM (kilobytes).
* ``l2_mb``                 — L2 total capacity (megabytes).
* ``num_l2_slices``         — Number of L2 partitions / slices.
* ``associativity_l2``      — L2 cache associativity.

The dataclass is intentionally a frozen, all-numeric struct.
There is NO arch one-hot field, NO ``sm_<n>`` enum, NO ``gpu_name``
string, NO ``vendor`` / ``sku`` / ``device_name`` identifier.
Whatever the architecture is, it must be encodable purely in the
six numeric values above; if a future architecture is not so
encodable, that's a model-redesign event, not a one-hot append.

This module is the input-contract anchor: the pooled head, when
trained, consumes only the fields declared here. The structural
contract is kept local to this module so the dataclass shape cannot
drift opaquely.

The ``from_cache_geometry`` adapter pulls the six values from the
output of
``baselines_harness.metrics.cache_geometry.resolve_cache_geometry()``.
The L2 capacity field is rescaled from bytes to megabytes
(integer division by 1024 * 1024) because the descriptor encodes
the magnitude that hardware decks publish; sub-megabyte L2 sizes
do not exist on the architectures of interest.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Final


# Canonical typed-reason vocabulary for the adapter's fail-closed
# paths. The pooled-head training pipeline catches these and emits
# the typed-unavailable abstention surface; the adapter does NOT
# silently coerce missing or non-positive fields into zero.
HARDWARE_DESCRIPTOR_TYPED_REASONS: Final[frozenset[str]] = frozenset({
    "hardware_descriptor_field_missing",
    "hardware_descriptor_field_non_positive",
    "hardware_descriptor_field_wrong_type",
})


class HardwareDescriptorError(Exception):
    """Raised when ``from_cache_geometry`` cannot build a valid
    ``HardwareDescriptor`` from the supplied geometry spec.
    Carries the canonical typed-reason string for downstream
    typed-unavailable bundle composition.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class HardwareDescriptor:
    """Numeric hardware-feature covariate vector for the pooled
    cross-arch calibration head.

    Frozen so consumers can use it as a dict key or set element
    when caching predictions per hardware target. All six fields
    are integer-typed; if a future hardware feature requires a
    rational or floating-point representation, the field must be
    added as a new integer-encoded ratio (e.g., milli-units) so
    the descriptor stays free of float-equality pitfalls in
    cache-identity hashing.
    """

    sector_size_bytes: int
    line_size_bytes: int
    l1_kb_per_sm: int
    l2_mb: int
    num_l2_slices: int
    associativity_l2: int


_BYTES_PER_MB: Final[int] = 1024 * 1024


def _require_positive_int(
    spec: dict[str, Any], key: str,
) -> int:
    """Pull ``spec[key]``, validate it is an integer-valued positive
    number, and return it as an ``int``. Raises
    :class:`HardwareDescriptorError` with a canonical typed-reason
    on missing / wrong-type / non-positive values.
    """
    if key not in spec or spec[key] is None:
        raise HardwareDescriptorError(
            "hardware_descriptor_field_missing",
            f"geometry_spec missing required field {key!r}",
        )
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise HardwareDescriptorError(
            "hardware_descriptor_field_wrong_type",
            f"geometry_spec[{key!r}]={value!r} is not int",
        )
    if value <= 0:
        raise HardwareDescriptorError(
            "hardware_descriptor_field_non_positive",
            f"geometry_spec[{key!r}]={value!r} is not positive",
        )
    return value


def from_cache_geometry(
    geometry_spec: dict[str, Any],
) -> HardwareDescriptor:
    """Build a :class:`HardwareDescriptor` from a
    ``resolve_cache_geometry()`` output dict.

    Expected input keys (each required, each positive int):

    * ``sector_size_bytes``
    * ``line_size_bytes``
    * ``l1_kb_per_sm``
    * ``l2_capacity_bytes``  (rescaled to ``l2_mb``)
    * ``num_l2_slices``
    * ``associativity_l2``

    The ``l2_capacity_bytes`` -> ``l2_mb`` rescaling uses integer
    division by 1024 * 1024 (one megabyte). Sub-megabyte L2 sizes
    are not used in the architectures of interest; if an input
    spec carries an ``l2_capacity_bytes`` value smaller than one
    megabyte the post-rescale value will be zero and the adapter
    will fail closed with ``hardware_descriptor_field_non_positive``.
    """
    sector_size_bytes = _require_positive_int(
        geometry_spec, "sector_size_bytes",
    )
    line_size_bytes = _require_positive_int(
        geometry_spec, "line_size_bytes",
    )
    l1_kb_per_sm = _require_positive_int(
        geometry_spec, "l1_kb_per_sm",
    )
    l2_capacity_bytes = _require_positive_int(
        geometry_spec, "l2_capacity_bytes",
    )
    num_l2_slices = _require_positive_int(
        geometry_spec, "num_l2_slices",
    )
    associativity_l2 = _require_positive_int(
        geometry_spec, "associativity_l2",
    )
    l2_mb = l2_capacity_bytes // _BYTES_PER_MB
    if l2_mb <= 0:
        raise HardwareDescriptorError(
            "hardware_descriptor_field_non_positive",
            (
                f"derived l2_mb={l2_mb} from "
                f"l2_capacity_bytes={l2_capacity_bytes}; "
                f"sub-megabyte L2 sizes are not supported"
            ),
        )
    return HardwareDescriptor(
        sector_size_bytes=sector_size_bytes,
        line_size_bytes=line_size_bytes,
        l1_kb_per_sm=l1_kb_per_sm,
        l2_mb=l2_mb,
        num_l2_slices=num_l2_slices,
        associativity_l2=associativity_l2,
    )


def descriptor_field_names() -> tuple[str, ...]:
    """Return the dataclass field names in declaration order.
    Public helper for the structural guard tests.
    """
    return tuple(f.name for f in fields(HardwareDescriptor))


__all__ = [
    "HARDWARE_DESCRIPTOR_TYPED_REASONS",
    "HardwareDescriptor",
    "HardwareDescriptorError",
    "descriptor_field_names",
    "from_cache_geometry",
]
