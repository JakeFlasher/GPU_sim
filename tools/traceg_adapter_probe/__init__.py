"""Bounded enhanced-trace to legacy Accel-Sim ``.traceg`` probe.

This package is intentionally outside the simulator crates.  It admits only a
one-CTA/one-warp common subset and fails closed for information that the legacy
trace grammar cannot represent.
"""

__all__ = ["AdapterRefusal", "RefusalCode", "adapt"]


def __getattr__(name: str):
    # Keep `python -m tools.traceg_adapter_probe.adapter` free of the runpy
    # double-import warning while retaining a small package API for callers.
    if name in __all__:
        from .adapter import AdapterRefusal, RefusalCode, adapt

        return {"AdapterRefusal": AdapterRefusal, "RefusalCode": RefusalCode, "adapt": adapt}[name]
    raise AttributeError(name)
