"""Runtime lookup API: slug → ladder state token.

Reads ``sm120_coverage_readiness_ladder_v1.toml`` once per
process (cached) and serves slug queries. Returns a member of
``_LADDER_STATES`` for any of the 235 corpus slugs; raises
``KeyError`` for unknown slugs (the manifest is total over the
corpus, so unknown-slug queries are caller bugs).
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_states import (
    _LADDER_STATES,
)


_DEFAULT_LADDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "sm120_coverage_readiness_ladder_v1.toml"
)


@lru_cache(maxsize=8)
def _load_ladder(toml_path: str) -> dict[str, str]:
    """Parse ``toml_path`` into a slug → state dict, caching the
    parsed result keyed by the path string.

    Returns:
        Frozen dict-style mapping (regular dict; treat as
        read-only). Every value is a ``_LADDER_STATES`` member;
        the loader asserts membership defensively so a tampered
        manifest fails at load time, not at lookup.
    """
    data = tomllib.loads(Path(toml_path).read_text("utf-8"))
    mapping: dict[str, str] = {}
    for entry in data.get("entries", []):
        slug = entry["slug"]
        state = entry["state"]
        if state not in _LADDER_STATES:
            raise ValueError(
                f"ladder entry {slug!r} has state {state!r} which "
                f"is not in _LADDER_STATES; manifest at {toml_path} "
                "appears corrupted or out of date"
            )
        mapping[slug] = state
    return mapping


def lookup_ladder_state(
    slug: str, *, ladder_path: Path | None = None
) -> str:
    """Return the highest-state token recorded for ``slug``.

    Args:
        slug: problem slug (e.g. ``problem_004_gemm_n128_k2048``).
        ladder_path: override the default manifest location (used
            by tests with planted fixtures).

    Returns:
        One of ``_LADDER_STATES`` members.

    Raises:
        KeyError: when ``slug`` is not in the manifest. The
            manifest is total over the 235 corpus, so this means
            either the slug is outside the corpus or the manifest
            is incomplete.
    """
    path = ladder_path if ladder_path is not None else _DEFAULT_LADDER_PATH
    mapping = _load_ladder(str(path))
    if slug not in mapping:
        raise KeyError(
            f"slug {slug!r} not present in ladder manifest at "
            f"{path}; corpus is expected to be total over the "
            f"235 problems"
        )
    return mapping[slug]


def all_ladder_slugs(*, ladder_path: Path | None = None) -> tuple[str, ...]:
    """Return the manifest's slugs in their canonical (ASCII-
    sorted) emission order."""
    path = ladder_path if ladder_path is not None else _DEFAULT_LADDER_PATH
    mapping = _load_ladder(str(path))
    return tuple(mapping.keys())
