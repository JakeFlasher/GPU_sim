"""ArchProfile — per-architecture hardware + policy cards read through one loader.

The merge centerpiece that turns hardcoded per-device constants into input data.
See ``docs/archprofile.md`` and ``loader.py``.
"""

from .loader import (
    ArchProfile,
    ArchProfileError,
    ArchContext,
    ResearchGap,
    RESEARCH_GAP,
    build_arch_family_for,
    is_research_gap,
    list_profiles,
    load_profile,
    resolve_arch_id,
    resolve_build_arch,
    resolve_context,
    resolve_sku,
)

__all__ = [
    "ArchProfile",
    "ArchProfileError",
    "ArchContext",
    "ResearchGap",
    "RESEARCH_GAP",
    "build_arch_family_for",
    "is_research_gap",
    "list_profiles",
    "load_profile",
    "resolve_arch_id",
    "resolve_build_arch",
    "resolve_context",
    "resolve_sku",
]
