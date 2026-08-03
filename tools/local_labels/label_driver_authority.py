"""Independent immutable authority for the canonical label driver."""
from __future__ import annotations

from pathlib import Path

LABEL_DRIVER_REL = Path("tools/local_labels/label_driver.sh")
LABEL_DRIVER_SHA256 = (
    "da64bb1f353d0531693d946b04428d8082d4cde9c861f475dfd00710be0f5788"
)
LABEL_DRIVER_MODE = 0o755
