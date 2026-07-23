"""Cache-model spec resolver + validator (task11 / the HRD-profiler clause).

Per the R16 task10 analysis artifact at
``.humanize/rlcr/2026-05-04_23-45-25/task10-ac6-analysis.md``
Sections 6 / 7 / 8.

Two distinct surfaces:

1. **Resolver** (``resolve_cache_model_spec``): given live device
   attributes (CUDA `l2CacheSize` etc.) plus an optional static
   device-card fallback, produce a fully-populated
   ``cache_model_spec`` dict. CUDA wins when present; static card
   is a fallback. CUDA-vs-static disagreement is non-failing
   corroboration evidence (the new round-added
   ``cache_model_capacity_static_card_disagreement`` typed
   reason).

2. **Validator** (``validate_declared_cache_model_spec``): given a
   caller-provided declared `cache_model_spec.capacity_bytes`
   (etc.), check it against the live CUDA value. A declared
   capacity that contradicts CUDA is REJECTED with the original-
   plan ``cache_model_capacity_inconsistent`` typed reason —
   this is the HRD-profiler-clause negative test at
   ``plans/nonroot_gpu_cache_observability_plan.md:202``.

The two surfaces have separate semantics: the resolver builds
the spec from live evidence and may attach corroboration
evidence; the validator gates a caller-supplied spec and rejects
contradictions. They MUST NOT collapse into a single typed
reason — Codex's R14 review caught that conflation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Producer-side typed reasons emitted by the cache-model spec
# resolver and validator. The full hrd_profile family lives in
# `profiler.HRD_PROFILE_TYPED_REASONS`; this subset is the
# cache-model-spec-specific slice and is imported from there to
# avoid drift.
CACHE_MODEL_SPEC_TYPED_REASONS: frozenset[str] = frozenset({
    "cache_model_capacity_missing",
    "cache_model_capacity_inconsistent",
    "cache_model_capacity_static_card_disagreement",
    "cache_model_geometry_missing",
    "cache_model_geometry_inconsistent",
    "cache_model_geometry_static_card_disagreement",
})


# Pinned defaults from R16 task10 artifact Section 6.
DEFAULT_SECTOR_SIZE_BYTES: int = 32
DEFAULT_LINE_SIZE_BYTES: int = 128
DEFAULT_SECTORS_PER_LINE: int = 4
DEFAULT_ASSOCIATIVITY_POLICY: str = "fully_associative_lru_approximation"
DEFAULT_FILL_POLICY: str = "write_allocate_on_global_load"
DEFAULT_WRITE_POLICY: str = "write_back_approximation"
DEFAULT_L1_BYPASS_CASES: tuple[str, ...] = (".cg", "atomic")
# A missing slice count is not equivalent to a one-slice cache.  The latter is
# a concrete model choice and must only be supplied by an identified caller.
# Keep the exported name for source compatibility, but make its value explicit
# rather than silently manufacturing geometry.
DEFAULT_NUM_L2_SLICES: int | None = None


class CacheModelGeometryUnavailable(ValueError):
    """Typed fail-closed result for unresolved cache geometry.

    A caller may still construct a schema placeholder for a typed-unavailable
    bundle, but it must never send that placeholder to the reuse-distance
    walker as a numeric prediction input.
    """

    TOKEN = "cache_model_geometry_missing"

    def __init__(self, detail: str, *, architecture: str | None = None) -> None:
        self.detail = detail
        self.architecture = architecture
        suffix = f" architecture={architecture!r}" if architecture else ""
        super().__init__(f"{self.TOKEN}:{suffix} {detail}")


class CacheModelSpecError(Exception):
    """Raised by ``validate_declared_cache_model_spec`` when a
    caller-provided declared spec contradicts CUDA device
    attributes. Carries the typed-reason string per the HRD-profiler clause."""

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


def _profile_geometry_for_architecture(
    architecture: str | None,
) -> tuple[int | None, int | None, str | None]:
    """Return ``(line_bytes, slices, detail)`` for an identified arch.

    The resolver is deliberately conservative: it may use a populated
    ArchProfile card, but it never substitutes sm_86 geometry for an unknown
    token or for a Blackwell research gap. ``detail`` is non-None when the
    caller supplied an architecture that cannot provide geometry.
    """
    if architecture is None:
        return None, None, "architecture identity was not supplied"
    try:
        from arch_profiles import ArchProfileError, is_research_gap, load_profile

        profile = load_profile(architecture)
    except Exception as exc:
        return None, None, f"unknown architecture {architecture!r}: {exc}"
    geometry = profile.l2_geometry
    if is_research_gap(geometry) or not isinstance(geometry, dict):
        return None, None, (
            f"L2 geometry is a research gap for architecture {profile.arch_id!r}"
        )
    raw_line = geometry.get("line_bytes")
    raw_slices = geometry.get("num_slices")
    line_bytes = (
        int(raw_line) if isinstance(raw_line, int) and raw_line > 0 else None
    )
    slices = (
        int(raw_slices)
        if isinstance(raw_slices, int) and not isinstance(raw_slices, bool)
        and raw_slices > 0
        else None
    )
    if line_bytes is None or slices is None:
        return None, None, (
            f"L2 geometry for architecture {profile.arch_id!r} is incomplete"
        )
    return line_bytes, slices, None


def resolve_cache_model_spec(
    *,
    cuda_l2_cache_size_bytes: int | None,
    static_card_l2_cache_size_bytes: int | None = None,
    cuda_line_size_bytes: int | None = None,
    static_card_line_size_bytes: int | None = None,
    cuda_num_l2_slices: int | None = None,
    static_card_num_l2_slices: int | None = None,
    architecture: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve the cache-model spec from live device attributes
    and optional static device-card fallbacks.

    Returns a ``(spec, evidence)`` pair. The spec is a fully-
    populated dict matching the schema's
    ``cache_model_spec`` block. The evidence is a list of
    informational-note records (``result_kind=info``) describing
    any CUDA-vs-static-card disagreements; the list is empty
    when CUDA + static agree (or when only one source is
    present). When the live device CANNOT supply capacity AND
    no static fallback is given, the spec carries
    ``capacity_source="typed_unavailable"`` and the evidence
    list contains a top-level cache_model_capacity_missing
    annotation that the bundle producer should escalate.

    Per Section 7's resolution order:
    1. CUDA wins when present and non-zero.
    2. Static-card fallback when CUDA is absent.
    3. Equal CUDA + static: CUDA wins, no disagreement note.
    4. Disagreement: CUDA wins, emit
       ``cache_model_capacity_static_card_disagreement`` info.

    `cuda_l2_cache_size_bytes = 0` is treated as absent (CUDA
    sometimes reports zero on virtual / pre-init contexts).
    """
    evidence: list[dict[str, Any]] = []

    # Architecture is an input identity, not an optional hint. When it is
    # supplied, use the card only as a source of measured geometry; an
    # unknown token is recorded as a typed research gap instead of borrowing
    # an Ampere constant.
    card_line_size, card_num_slices, architecture_detail = (
        _profile_geometry_for_architecture(architecture)
        if architecture is not None
        else (None, None, None)
    )
    if architecture is not None and architecture_detail is not None:
        evidence.append({
            "result_kind": "typed_unavailable",
            "typed_reason": "cache_model_geometry_missing",
            "scope": "bundle",
        })

    capacity_bytes, capacity_source, cap_evidence = _resolve_field(
        cuda_value=cuda_l2_cache_size_bytes,
        static_card_value=static_card_l2_cache_size_bytes,
        disagreement_note_kind=(
            "cache_model_capacity_static_card_disagreement"
        ),
    )
    evidence.extend(cap_evidence)
    if capacity_bytes is None:
        evidence.append({
            "result_kind": "typed_unavailable",
            "typed_reason": "cache_model_capacity_missing",
            "scope": "bundle",
        })

    line_size_bytes, line_source, geom_evidence = _resolve_field(
        cuda_value=cuda_line_size_bytes,
        static_card_value=(
            static_card_line_size_bytes
            if static_card_line_size_bytes is not None
            else card_line_size
        ),
        disagreement_note_kind=(
            "cache_model_geometry_static_card_disagreement"
        ),
        constant_default=DEFAULT_LINE_SIZE_BYTES,
    )
    evidence.extend(geom_evidence)

    num_l2_slices, num_l2_slices_source, slices_evidence = _resolve_field(
        cuda_value=cuda_num_l2_slices,
        static_card_value=(
            static_card_num_l2_slices
            if static_card_num_l2_slices is not None
            else card_num_slices
        ),
        disagreement_note_kind=(
            "cache_model_geometry_static_card_disagreement"
        ),
        # There is intentionally no constant default here. A one-slice cache
        # is a model assumption, not an identity-safe fallback.
        constant_default=None,
    )
    evidence.extend(slices_evidence)

    if num_l2_slices is None:
        # Keep the schema-facing source explicit. The producer will replace
        # the absent value with a non-semantic placeholder only on the typed
        # unavailable arm; it must never send this result to the walker.
        num_l2_slices_source = "typed_unavailable"
        evidence.append({
            "result_kind": "typed_unavailable",
            "typed_reason": "cache_model_geometry_missing",
            "scope": "bundle",
        })

    geometry_source = (
        "typed_unavailable"
        if num_l2_slices is None or line_size_bytes is None
        else (
            line_source
            if line_source == num_l2_slices_source
            else "cuda_device_prop"
            if "cuda_device_prop" in (line_source, num_l2_slices_source)
            else "static_device_card"
        )
    )

    spec: dict[str, Any] = {
        "capacity_bytes": capacity_bytes if capacity_bytes is not None else 0,
        "sector_size_bytes": DEFAULT_SECTOR_SIZE_BYTES,
        "line_size_bytes": (
            line_size_bytes if line_size_bytes is not None
            else DEFAULT_LINE_SIZE_BYTES
        ),
        "sectors_per_line": DEFAULT_SECTORS_PER_LINE,
        "associativity_policy": DEFAULT_ASSOCIATIVITY_POLICY,
        "fill_policy": DEFAULT_FILL_POLICY,
        "write_policy": DEFAULT_WRITE_POLICY,
        "l1_bypass_cases": list(DEFAULT_L1_BYPASS_CASES),
        "capacity_source": capacity_source,
        "geometry_source": geometry_source,
        "num_l2_slices": num_l2_slices,
        "num_l2_slices_source": num_l2_slices_source,
    }
    return spec, evidence


def _resolve_field(
    *,
    cuda_value: int | None,
    static_card_value: int | None,
    disagreement_note_kind: str,
    constant_default: int | None = None,
) -> tuple[int | None, str, list[dict[str, Any]]]:
    """Apply the CUDA-vs-static-card resolution rule to a single
    integer field. Returns ``(resolved_value, source_label,
    evidence)``."""
    cuda_present = cuda_value is not None and cuda_value > 0
    static_present = static_card_value is not None
    evidence: list[dict[str, Any]] = []
    if cuda_present:
        assert cuda_value is not None  # narrowed by cuda_present
        if static_present and static_card_value != cuda_value:
            evidence.append({
                "result_kind": "info",
                "note_kind": disagreement_note_kind,
                "cuda_value": int(cuda_value),
                "static_card_value": int(static_card_value)
                if static_card_value is not None else None,
            })
        return int(cuda_value), "cuda_device_prop", evidence
    if static_present:
        assert static_card_value is not None
        return int(static_card_value), "static_device_card", evidence
    if constant_default is not None:
        return int(constant_default), "constant", evidence
    return None, "typed_unavailable", evidence


def validate_declared_cache_model_spec(
    declared: dict[str, Any],
    *,
    cuda_l2_cache_size_bytes: int | None,
    cuda_line_size_bytes: int | None = None,
) -> None:
    """Validate a caller-provided declared `cache_model_spec`
    against the live CUDA device attributes. The original-plan
    HRD-profiler-clause negative test
    (`plans/nonroep_gpu_cache_observability_plan.md:202`)
    requires the validator to REJECT a declared spec whose
    capacity contradicts the live CUDA `l2CacheSize`.

    Raises ``CacheModelSpecError`` with one of these typed
    reasons:

    - ``cache_model_capacity_missing``: declared spec lacks a
      ``capacity_bytes`` field (or is zero) AND CUDA is also
      absent.
    - ``cache_model_capacity_inconsistent``: declared
      ``capacity_bytes`` contradicts CUDA's ``l2CacheSize``
      (the caller asserted a value the live device does not
      confirm).
    - ``cache_model_geometry_inconsistent``: same shape for
      ``line_size_bytes``.

    The corroboration arm (CUDA vs. static-card disagreement
    during runtime resolution) is NOT this validator's
    responsibility — that arm is the resolver's
    ``cache_model_capacity_static_card_disagreement`` info
    evidence.
    """
    declared_capacity = declared.get("capacity_bytes")
    if (
        declared_capacity is None or declared_capacity == 0
    ) and (
        cuda_l2_cache_size_bytes is None
        or cuda_l2_cache_size_bytes == 0
    ):
        raise CacheModelSpecError(
            "cache_model_capacity_missing",
            f"declared capacity_bytes={declared_capacity} and CUDA "
            f"l2CacheSize={cuda_l2_cache_size_bytes}; both absent",
        )
    if (
        declared_capacity is not None
        and declared_capacity > 0
        and cuda_l2_cache_size_bytes is not None
        and cuda_l2_cache_size_bytes > 0
        and int(declared_capacity) != int(cuda_l2_cache_size_bytes)
    ):
        raise CacheModelSpecError(
            "cache_model_capacity_inconsistent",
            f"declared capacity_bytes={declared_capacity} contradicts "
            f"CUDA l2CacheSize={cuda_l2_cache_size_bytes}",
        )
    declared_line = declared.get("line_size_bytes")
    if (
        declared_line is not None
        and cuda_line_size_bytes is not None
        and cuda_line_size_bytes > 0
        and int(declared_line) != int(cuda_line_size_bytes)
    ):
        raise CacheModelSpecError(
            "cache_model_geometry_inconsistent",
            f"declared line_size_bytes={declared_line} contradicts "
            f"CUDA line size={cuda_line_size_bytes}",
        )


__all__ = [
    "CacheModelSpecError",
    "CACHE_MODEL_SPEC_TYPED_REASONS",
    "DEFAULT_SECTOR_SIZE_BYTES",
    "DEFAULT_LINE_SIZE_BYTES",
    "DEFAULT_SECTORS_PER_LINE",
    "DEFAULT_ASSOCIATIVITY_POLICY",
    "DEFAULT_FILL_POLICY",
    "DEFAULT_WRITE_POLICY",
    "DEFAULT_L1_BYPASS_CASES",
    "resolve_cache_model_spec",
    "validate_declared_cache_model_spec",
]
