"""Import-safe tombstone for retired label-trained cache calibration.

This package initializer publishes only the fail-closed retirement contract and
never imports an executable legacy producer. Historical modules are retained as
source evidence rather than re-exported as an active package API.
"""

from .retirement import (
    LEGACY_NCU_CALIBRATION_RETIRED,
    LegacyCalibrationRetiredError,
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

__all__ = [
    "LEGACY_NCU_CALIBRATION_RETIRED",
    "LegacyCalibrationRetiredError",
    "refuse_legacy_ncu_calibration",
    "retired_legacy_ncu_calibration",
]
