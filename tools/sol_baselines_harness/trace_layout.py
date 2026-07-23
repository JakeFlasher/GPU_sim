"""Cross-arch trace-layout contract helpers.

Single source of truth for `.harness_traces/` bundle-name parsing,
arch derivation, default bundle-dir resolution for fresh emissions,
and the overwrite-into-locked-sm_120-bundle guard.

Implements the DEC-B-RESOLVED sibling-dir layout documented at
`scripts/baselines_harness/calibration/cache_prediction/
coverage_expansion_trace_layout_contract.md`.

Three accepted bundle-name forms:

  ``<slug>__<lang>/``            legacy un-suffixed (86 sm_120 R0-R23 bundles)
  ``<slug>__<lang>__sm86/``      new sm_86 sibling-dir
  ``<slug>__<lang>__sm120/``     reserved future sm_120 re-runs

Public API:

* ``BUNDLE_NAME_RE`` — canonical regex.
* ``TRACE_LAYOUT_TYPED_REASONS`` — closed-vocabulary set of typed reasons.
* ``TraceLayoutViolation`` — raised by the guards with a ``typed_reason``.
* ``parse_bundle_name(name)`` — returns ``(slug, lang, arch_suffix_or_None)``.
* ``arch_suffix_from_host_kind(host_kind)`` — host-to-suffix mapping.
* ``default_bundle_dir(traces_root, slug, language, host_kind=None)`` —
  resolves the canonical default bundle path for a fresh emission.
* ``assert_no_sm120_overwrite(bundle_dir, host_kind)`` — fail-closed
  guard for a sm_86 writer about to clobber a locked sm_120 bundle.
* ``evidence_key_for_bundle(bundle_dir, manifest_dict=None)`` —
  returns the ``(slug, lang, arch)`` tuple used by consistency
  consumers.
* ``effective_arch_for_bundle(bundle_name, manifest_dict=None)`` —
  shorthand returning just the arch tag.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final, Optional, Tuple


BUNDLE_NAME_RE: Final[re.Pattern] = re.compile(
    # The slug character class allows uppercase letters: the canonical
    # harness-facing slug for the three truncation-case problems is the
    # benchmark dir truncated at the first non-[A-Za-z0-9_] char, which
    # preserves uppercase (e.g. ``problem_079_ImageNet_83``). The canonical
    # form never contains ``.`` / ``-`` (those are truncated away upstream
    # by sm86_port_scope.canonical_slug), so the bundle name stays parseable.
    r"^(?P<slug>problem_\d+_[A-Za-z0-9_]+)"
    r"__(?P<lang>cuda_cpp|cutlass)"
    # Optional arch suffix, with an optional ``_cov`` coverage-variant
    # marker: the design contract emits a fresh ``…__sm86_cov`` provenance-valid bundle for
    # a cell whose canonical ``…__sm86`` path is occupied by a frozen pilot
    # bundle; ``arch`` still resolves to the base arch (sm86) for the
    # overwrite guard and host routing.
    r"(?:__(?P<arch>sm_?86|sm_?120)(?P<coverage>_cov)?)?$"
)


TRACE_LAYOUT_TYPED_REASONS: Final[frozenset] = frozenset({
    "trace_layout_overwrite_sm120_evidence",
    "evidence_label_arch_dimension_missing",
    "trace_layout_bundle_name_unparseable",
    "trace_layout_arch_suffix_manifest_disagreement",
})


class TraceLayoutViolation(Exception):
    """Raised when a bundle-dir read or write would violate the
    DEC-B sibling-dir contract.

    The ``typed_reason`` attribute carries a member of
    ``TRACE_LAYOUT_TYPED_REASONS`` for closed-vocabulary discipline.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        if typed_reason not in TRACE_LAYOUT_TYPED_REASONS:
            raise ValueError(
                f"unknown trace-layout typed_reason {typed_reason!r}; "
                f"valid: {sorted(TRACE_LAYOUT_TYPED_REASONS)}"
            )
        self.typed_reason = typed_reason
        super().__init__(f"{typed_reason}: {message}")


def parse_bundle_name(
    name: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Return ``(slug, language, arch_suffix_or_None)`` or ``None`` if
    the name does not match the contract regex."""
    m = BUNDLE_NAME_RE.match(name)
    if m is None:
        return None
    return (m.group("slug"), m.group("lang"), m.group("arch"))


# host.kind -> arch_suffix for the default bundle-dir resolver.
# local_sm_120 maps to None (the legacy un-suffixed path is the
# locked canonical for sm_120 evidence; no new sm_120 writes go to
# __sm120 under this plan).
_HOST_KIND_TO_ARCH_SUFFIX: Final[dict] = {
    "local_sm_8_6": "sm86",
    "local_sm_120": None,
    "b200_sm_100a": None,
}


def arch_suffix_from_host_kind(host_kind: Optional[str]) -> Optional[str]:
    """Map a host.kind token to a bundle-name arch suffix.

    Returns ``None`` when the host should use the legacy un-suffixed
    canonical path (sm_120, B200, or unknown / ``None`` input).
    """
    if host_kind is None:
        return None
    return _HOST_KIND_TO_ARCH_SUFFIX.get(host_kind)


def default_bundle_dir(
    traces_root: Path,
    slug: str,
    language: str,
    host_kind: Optional[str] = None,
) -> Path:
    """Resolve the default bundle directory for a fresh emission.

    Routing per DEC-B sibling-dir contract:

    * ``host_kind="local_sm_8_6"`` (the active 3090 Ti host) -> the
      arch-qualified sibling form ``<slug>__<lang>__sm86/``.
    * ``host_kind="local_sm_120"`` (a future 5060 host) -> legacy
      un-suffixed ``<slug>__<lang>/``. The locked R0-R23 sm_120
      bundles already occupy these paths; the design contract byte-immutability
      forbids any write that mutates them.
    * ``host_kind="b200_sm_100a"`` / ``None`` / unknown -> legacy
      un-suffixed ``<slug>__<lang>/``.

    The locked sm_120 R0-R23 bundles continue to be read at their
    un-suffixed canonical paths. The host-aware default only affects
    fresh sm_86 writes — those go to ``__sm86`` so the design contract's locked set
    cannot be clobbered by a default-path emission.
    """
    suffix = arch_suffix_from_host_kind(host_kind)
    if suffix is None:
        return traces_root / f"{slug}__{language}"
    return traces_root / f"{slug}__{language}__{suffix}"


def assert_no_sm120_overwrite(
    bundle_dir: Path, host_kind: Optional[str]
) -> None:
    """Fail closed if a sm_86 host would clobber a locked sm_120
    canonical bundle by writing to ``bundle_dir``.

    The guard fires only when ALL three conditions hold:

    1. ``host_kind == "local_sm_8_6"`` (the active sm_86 host).
    2. ``bundle_dir.name`` parses as the legacy un-suffixed form
       (no ``__sm86`` / ``__sm120`` suffix).
    3. ``bundle_dir/manifest.json`` exists, parses, and declares
       ``host.kind == "local_sm_120"``.

    Otherwise no-op. The intended call site is just before any
    sm_86 fanout writes its manifest into an operator-supplied
    explicit ``--bundle-dir`` target; the default resolver already
    routes sm_86 hosts to ``__sm86`` paths so the guard does NOT
    fire on default-path writes.

    Raises ``TraceLayoutViolation`` with typed reason
    ``trace_layout_overwrite_sm120_evidence``.
    """
    if host_kind != "local_sm_8_6":
        return
    parsed = parse_bundle_name(bundle_dir.name)
    if parsed is None:
        return
    _slug, _lang, suffix = parsed
    if suffix is not None:
        return
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        m = json.loads(manifest_path.read_text())
    except Exception:
        return
    existing_kind = (m.get("host", {}) or {}).get("kind")
    if existing_kind == "local_sm_120":
        suggested = bundle_dir.parent / f"{bundle_dir.name}__sm86"
        raise TraceLayoutViolation(
            "trace_layout_overwrite_sm120_evidence",
            (
                f"sm_86 host host_kind={host_kind!r} would write into "
                f"{bundle_dir} which currently holds a sm_120 "
                f"manifest (host.kind={existing_kind!r}). Use the "
                f"arch-qualified sibling {suggested} instead."
            ),
        )


def evidence_key_for_bundle(
    bundle_dir: Path, manifest_dict: Optional[dict] = None
) -> Tuple[str, str, str]:
    """Return the ``(slug, language, arch)`` evidence key for a bundle.

    Arch derivation:

    * If the bundle name carries a ``__sm86`` / ``__sm120`` suffix,
      the suffix wins.
    * Otherwise (legacy un-suffixed form), the arch is derived from
      ``manifest.host.kind`` — ``local_sm_120`` -> ``"sm120"``,
      ``local_sm_8_6`` -> ``"sm86"``.
    * If the suffix and the manifest disagree, raises
      ``trace_layout_arch_suffix_manifest_disagreement``.
    * If neither is resolvable, raises
      ``evidence_label_arch_dimension_missing``.
    * If the name does not parse at all, raises
      ``trace_layout_bundle_name_unparseable``.
    """
    parsed = parse_bundle_name(bundle_dir.name)
    if parsed is None:
        raise TraceLayoutViolation(
            "trace_layout_bundle_name_unparseable",
            f"bundle name {bundle_dir.name!r} does not match "
            f"BUNDLE_NAME_RE={BUNDLE_NAME_RE.pattern!r}",
        )
    slug, lang, suffix = parsed

    if manifest_dict is None:
        manifest_path = bundle_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest_dict = json.loads(manifest_path.read_text())
            except Exception:
                manifest_dict = None

    manifest_arch: Optional[str] = None
    if manifest_dict is not None:
        host_kind = (manifest_dict.get("host", {}) or {}).get("kind")
        if host_kind == "local_sm_8_6":
            manifest_arch = "sm86"
        elif host_kind == "local_sm_120":
            manifest_arch = "sm120"

    if suffix is not None and manifest_arch is not None and suffix != manifest_arch:
        raise TraceLayoutViolation(
            "trace_layout_arch_suffix_manifest_disagreement",
            f"bundle {bundle_dir.name!r}: name suffix={suffix!r} "
            f"disagrees with manifest host.kind-derived arch "
            f"{manifest_arch!r}",
        )

    arch = suffix if suffix is not None else manifest_arch
    if arch is None:
        raise TraceLayoutViolation(
            "evidence_label_arch_dimension_missing",
            f"bundle {bundle_dir.name!r} has no arch suffix AND its "
            f"manifest does not declare host.kind; arch dimension "
            f"is unresolvable",
        )
    return (slug, lang, arch)


def effective_arch_for_bundle(
    bundle_name: str, manifest_dict: Optional[dict] = None
) -> Optional[str]:
    """Convenience: return just the arch tag for a bundle, or None
    if unresolvable. Never raises; callers that need the typed-reason
    failure modes call ``evidence_key_for_bundle`` instead."""
    parsed = parse_bundle_name(bundle_name)
    if parsed is None:
        return None
    _slug, _lang, suffix = parsed
    if suffix is not None:
        return suffix
    if manifest_dict is None:
        return None
    host_kind = (manifest_dict.get("host", {}) or {}).get("kind")
    if host_kind == "local_sm_8_6":
        return "sm86"
    if host_kind == "local_sm_120":
        return "sm120"
    return None
