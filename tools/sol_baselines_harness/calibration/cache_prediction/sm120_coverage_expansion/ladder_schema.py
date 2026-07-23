"""Schema definitions for the sm_120 coverage-readiness ladder
manifest.

Uses plain ``dataclasses`` (no Pydantic) to match the codebase's
calibration-prediction style (see ``label_parser.py``,
``calibration_validation_split.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

LADDER_SCHEMA_VERSION: str = "1.0"
LADDER_TOTAL_PROBLEMS: int = 235
LADDER_HOST_ARCH: str = "sm_120"

# Required meta-block fields the QA validator expects in every
# emitted ladder TOML.
REQUIRED_META_FIELDS: tuple[str, ...] = (
    "schema_version",
    "host_arch",
    "captured_at_utc",
    "total_problems",
    "corpus_sha256",
)


@dataclass(frozen=True, slots=True)
class LadderEntry:
    """One per-problem row in the ladder manifest.

    Fields:
        slug: full markdown stem (e.g.
            ``problem_004_gemm_n128_k2048``).
        state: one of ``_LADDER_STATES`` (validated by the QA
            module, not in the dataclass — keeps the dataclass
            cheap and re-usable for planted-mutation negatives).
    """

    slug: str
    state: str


@dataclass(frozen=True, slots=True)
class LadderManifestMeta:
    """Meta block of the ladder TOML.

    ``corpus_sha256`` is a SHA-256 over the deterministically-
    sorted concatenation of the 235 problem-spec slugs (newline-
    separated, UTF-8). The deterministic re-emission test in
    ``test_sm120_coverage_readiness_ladder.py`` asserts the
    corpus SHA is stable across emissions.
    """

    schema_version: str
    host_arch: str
    captured_at_utc: str
    total_problems: int
    corpus_sha256: str


@dataclass(frozen=True, slots=True)
class LadderManifest:
    """Full ladder manifest: meta + ordered entries.

    Entries MUST be sorted ASCII by ``slug`` (the QA validator
    rejects non-sorted manifests with
    ``ladder_ordering_violation``).
    """

    meta: LadderManifestMeta
    entries: tuple[LadderEntry, ...] = field(default_factory=tuple)
