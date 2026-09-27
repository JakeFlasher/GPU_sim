"""Bounded enhanced-trace to legacy Accel-Sim ``.traceg`` probe.

This package is intentionally outside the simulator crates.  It admits only a
one-CTA/one-warp common subset and fails closed for information that the legacy
trace grammar cannot represent.
"""

from .adapter import AdapterRefusal, RefusalCode, adapt

__all__ = ["AdapterRefusal", "RefusalCode", "adapt"]
