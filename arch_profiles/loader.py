"""ArchProfile loader + canonical arch-id / SKU resolver.

This is the merge centerpiece: the single source of truth that turns the
per-architecture constants — historically hardcoded in per-device dicts
(``_SM_COUNT_BY_ARCH``, ``_GRIDDIM_PENALTY_ALPHA``, ``_STATIC_CARD_L2_BYTES``,
``_BARE_LDG_E_DEFAULTS_TO_CA``, the NVML reference operating point, the L1/L2
geometry cards) — into **input data cards** read through one loader.

Design (grounded in the seam research, esp. Q9 "token vocabulary"):

* The canonical architecture identity is the **bare ``sm_XX`` token** (e.g.
  ``sm_86``, ``sm_120``). This matches ``provenance_audit._canonicalize_request_arch``
  which already declares the bare form canonical, and keeps the frozen
  ``label_admitted.json`` arch enum ``{sm_86, sm_120}`` and the 30-cell sm_120
  cohort byte-stable.
* A device-tier **SKU** (``rtx_3090_ti``, ``rtx_5060_laptop``) disambiguates
  hardware facts that differ between two cards sharing a compute capability
  (a bare ``sm_120`` cannot tell a GB206 laptop from a GB202 workstation — the
  ``sm_120-laptop`` alias exists precisely to prevent that conflation).
* ``build_arch_family`` (a compile-time concern) is kept SEPARATE from the
  deploy/collection ``arch_id`` (Q5): a fatbin may legitimately carry server
  slices ``sm_90a`` / ``sm_120a`` that are never valid deploy arches.

The loader intentionally performs **zero** behaviour change for sm_86: every
sm_86 card field equals the value hardcoded in the live source today, and
``tests/test_arch_profile_lockstep.py`` asserts that byte-for-byte. The only
deliberate *correction* is the sm_120 SM count (card = measured 26, superseding
the stale guessed 50 still sitting in ``cache_predictor._SM_COUNT_BY_ARCH``).

Fields whose real value was never captured on the local Blackwell part
(sm_120 L1 geometry, NVML anchor, bare-LDG default) are represented as an
explicit ``{"status": "research_gap", ...}`` marker rather than a silent
default — see ``OPEN_RESEARCH_CHALLENGES.md``. Reading such a field returns the
marker; callers must fail closed, never fall back to sm_86 values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_CARDS_DIR = _HERE / "cards"
_ALIASES_PATH = _HERE / "aliases.json"
_BUILD_FAMILIES_PATH = _HERE / "build_arch_families.json"

# Sentinel returned for a field the local hardware never produced.
RESEARCH_GAP = "research_gap"


class ArchProfileError(ValueError):
    """Raised on an unresolvable token, an unknown card, or a rejected SKU."""


class ResearchGap(dict):
    """A typed-absent field. Truthy-safe: ``bool(gap)`` is False.

    A card field that was never captured on local silicon is stored as
    ``{"status": "research_gap", "reason": ..., "challenge": ...}``. Consumers
    should check :func:`is_research_gap` and fail closed rather than substitute
    another arch's value.
    """

    def __bool__(self) -> bool:  # so ``if profile.l1_geometry:`` is False on a gap
        return False


def is_research_gap(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") == RESEARCH_GAP


@dataclass(frozen=True)
class ArchProfile:
    """One loaded architecture card. ``raw`` holds the full JSON document."""

    arch_id: str
    sku: str
    raw: dict[str, Any] = field(repr=False)

    # --- convenience accessors over the raw card ------------------------------
    @property
    def compute_capability(self) -> tuple[int, int]:
        cc = self.raw["compute_capability"]
        return (int(cc[0]), int(cc[1]))

    @property
    def sm_count(self) -> int | ResearchGap:
        return _gap_or(self.raw.get("sm_count"))

    @property
    def griddim_penalty_alpha(self) -> float | ResearchGap:
        return _gap_or(self.raw.get("griddim_penalty_alpha"))

    @property
    def l2_geometry(self) -> dict[str, Any] | ResearchGap:
        return _gap_or(self.raw.get("l2_geometry"))

    @property
    def l1_geometry(self) -> dict[str, Any] | ResearchGap:
        return _gap_or(self.raw.get("l1_geometry"))

    @property
    def bare_ldg_e_default(self) -> str:
        # "ca" | "bypass" | "unknown_typed_unavailable"
        return self.raw.get("bare_ldg_e_default", "unknown_typed_unavailable")

    @property
    def cache_layers(self) -> list[str]:
        return list(self.raw.get("cache_layers", ["L2"]))

    @property
    def nvml_reference_operating_point(self) -> dict[str, Any] | ResearchGap:
        return _gap_or(self.raw.get("nvml_reference_operating_point"))

    @property
    def build_arch_family(self) -> list[str]:
        return list(self.raw.get("build_arch_family", [self.arch_id]))

    @property
    def evidence_status(self) -> str:
        return self.raw.get("evidence_status", "Offline-verified")

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def _gap_or(value: Any) -> Any:
    if is_research_gap(value):
        return ResearchGap(value)
    return value


@dataclass(frozen=True)
class ArchContext:
    """Validated deployment/SKU/build identity for production consumers.

    ``ArchProfile`` remains the card data source; this wrapper prevents a
    caller from resolving a deployment architecture from a bare compute
    capability and then silently selecting a different SKU or build slice.
    """

    arch_id: str
    sku: str
    build_arch: str
    profile: ArchProfile

    def require(self, field: str) -> Any:
        """Return a card field or raise on a typed research gap."""
        value = self.profile.get(field)
        if value is None or is_research_gap(value):
            reason = value.get("reason", "") if is_research_gap(value) else "field absent"
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: capability {field!r} unavailable: {reason}"
            )
        return value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def list_profiles() -> list[str]:
    """Return the sorted ``arch_id`` values of all installed cards."""
    return sorted(_load_json(p)["arch_id"] for p in _CARDS_DIR.glob("*.json"))


def _card_path_for(arch_id: str, sku: str | None = None) -> Path:
    """Find the card file whose (arch_id[, sku]) matches."""
    matches = []
    for p in sorted(_CARDS_DIR.glob("*.json")):
        card = _load_json(p)
        if card["arch_id"] != arch_id:
            continue
        if sku is not None and card.get("sku") != sku:
            continue
        matches.append((p, card))
    if not matches:
        raise ArchProfileError(
            f"no ArchProfile card for arch_id={arch_id!r} sku={sku!r} "
            f"(installed: {list_profiles()})"
        )
    if len(matches) > 1:
        skus = [c.get("sku") for _, c in matches]
        raise ArchProfileError(
            f"arch_id={arch_id!r} is ambiguous across SKUs {skus}; pass sku=..."
        )
    return matches[0][0]


def load_profile(token: str, sku: str | None = None) -> ArchProfile:
    """Load the card for any arch spelling / device name / SKU token.

    ``token`` may be a canonical arch_id, any alias, a device-name substring,
    or a SKU. The canonical arch_id is resolved first via :func:`resolve_arch_id`.
    """
    arch_id = resolve_arch_id(token)
    if sku is None:
        sku = resolve_sku(token, strict=False)
    card = _load_json(_card_path_for(arch_id, sku))
    return ArchProfile(arch_id=card["arch_id"], sku=card.get("sku", ""), raw=card)


def resolve_build_arch(token: str, *, deploy_arch: str | None = None) -> str:
    """Validate a compile/fatbin token without treating it as deployment identity."""
    raw = str(token).strip().lower()
    families = _load_json(_BUILD_FAMILIES_PATH)["compute_capability_to_build_family"]
    known = {build for family in families.values() for build in family}
    if raw not in known:
        raise ArchProfileError(f"unknown build architecture token {token!r}")
    if deploy_arch is not None:
        arch = resolve_arch_id(deploy_arch)
        profile = load_profile(arch)
        if raw not in profile.build_arch_family:
            raise ArchProfileError(
                f"build token {raw!r} is not in {arch} family {profile.build_arch_family!r}"
            )
    return raw


def resolve_context(
    token: str | tuple[int, int],
    *,
    sku: str | None,
    build_arch: str,
) -> ArchContext:
    """Resolve and validate a complete deployment/SKU/build identity.

    The SKU and build token are intentionally mandatory. A bare ``(12, 0)``
    or ``sm_120`` token is not enough to authorize a production input because
    multiple SKUs may share an ISA and build-only tokens are not deployment
    identities.
    """
    raw = str(token).strip().lower() if not isinstance(token, (tuple, list)) else None
    build_only = set(_load_json(_BUILD_FAMILIES_PATH).get("build_only_tokens", []))
    if raw in build_only:
        raise ArchProfileError(
            f"build token {token!r} cannot be used as deployment identity"
        )
    if sku is None:
        raise ArchProfileError("complete architecture identity requires an explicit SKU")
    if isinstance(token, (tuple, list)):
        # Do not let a CC-only request become a card selection by today's
        # single-card inventory; the ambiguity must remain explicit.
        arch_id = resolve_arch_id(token)
    else:
        arch_id = resolve_arch_id(token)
    canonical_sku = resolve_sku(sku, strict=True)
    if canonical_sku != sku.strip().lower():
        raise ArchProfileError(
            f"SKU token {sku!r} is not canonical; use {canonical_sku!r}"
        )
    token_sku = resolve_sku(token, strict=False)
    if token_sku is not None and token_sku != canonical_sku:
        raise ArchProfileError(
            f"deployment token {token!r} identifies SKU {token_sku!r}, not {canonical_sku!r}"
        )
    sku_cards = [
        _load_json(path)
        for path in sorted(_CARDS_DIR.glob("*.json"))
        if _load_json(path).get("sku") == canonical_sku
    ]
    if not sku_cards:
        raise ArchProfileError(f"no ArchProfile card for SKU {canonical_sku!r}")
    sku_arches = {card["arch_id"] for card in sku_cards}
    if arch_id not in sku_arches:
        raise ArchProfileError(
            f"SKU {canonical_sku!r} belongs to architecture(s) {sorted(sku_arches)!r}, "
            f"not {arch_id!r}"
        )
    profile = load_profile(arch_id, sku=canonical_sku)
    build = resolve_build_arch(build_arch, deploy_arch=arch_id)
    if profile.sku != canonical_sku:
        raise ArchProfileError(
            f"card SKU {profile.sku!r} disagrees with requested {canonical_sku!r}"
        )
    return ArchContext(
        arch_id=arch_id,
        sku=canonical_sku,
        build_arch=build,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Resolution (the alias landmine fix, Q9)
# ---------------------------------------------------------------------------
def _aliases() -> dict[str, Any]:
    return _load_json(_ALIASES_PATH)


def resolve_arch_id(token: str | tuple[int, int]) -> str:
    """Normalise any of the 8 coexisting spellings → canonical bare ``sm_XX``.

    Accepts: canonical ids (``sm_86``), device/capability aliases
    (``sm_120-laptop``), build tokens (``sm_120a``, ``sm_120f``), a
    ``(major, minor)`` compute-capability tuple, or a device-name substring
    ("NVIDIA GeForce RTX 5060 Laptop GPU"). Rejected server/workstation SKUs
    (e.g. RTX PRO 6000) raise :class:`ArchProfileError` — they are not local
    deploy arches (Q9 / preflight policy).
    """
    aliases = _aliases()

    # (major, minor) compute capability
    if isinstance(token, (tuple, list)):
        cc = f"{int(token[0])}.{int(token[1])}"
        arch = aliases["compute_capability_to_arch_id"].get(cc)
        if arch is None:
            raise ArchProfileError(f"no arch_id for compute capability {cc}")
        return arch

    raw = str(token).strip()
    low = raw.lower()

    # explicit rejections first (server/workstation cards must not masquerade
    # as a local deploy arch)
    for bad in aliases.get("rejected_device_substrings", []):
        if bad.lower() in low:
            raise ArchProfileError(
                f"device {raw!r} matches rejected substring {bad!r}: it is a "
                f"non-local (server/workstation) card without an ArchProfile"
            )

    # exact token → canonical id (covers sm_86, sm_120, sm_120-laptop, sm_120a…)
    exact = aliases["token_to_arch_id"]
    if raw in exact:
        return exact[raw]
    if low in exact:
        return exact[low]

    # device-name substring → canonical id
    for sub, arch in aliases["device_substring_to_arch_id"].items():
        if sub.lower() in low:
            return arch

    raise ArchProfileError(
        f"cannot resolve arch_id from token {raw!r}; known tokens: "
        f"{sorted(exact)}"
    )


def resolve_sku(token: str | tuple[int, int], *, strict: bool = True) -> str | None:
    """Resolve a device-tier SKU (``rtx_3090_ti`` / ``rtx_5060_laptop``).

    A bare arch_id or compute capability cannot pin a SKU on its own when a
    capability hosts multiple cards; ``strict=False`` returns None instead of
    raising so ``load_profile`` can fall back to the sole card for an arch_id.
    """
    aliases = _aliases()
    if isinstance(token, (tuple, list)):
        if strict:
            raise ArchProfileError("compute capability alone cannot pin a SKU")
        return None
    low = str(token).strip().lower()

    if low in aliases["sku_tokens"]:
        return low
    for sub, sku in aliases["device_substring_to_sku"].items():
        if sub.lower() in low:
            return sku
    if strict:
        raise ArchProfileError(f"cannot resolve SKU from token {token!r}")
    return None


def build_arch_family_for(cc: tuple[int, int]) -> list[str]:
    """Return the fatbin build-arch family for a compute capability (Q5).

    NB: this is a *build*-time vocabulary and may include server arches
    (``sm_90a``, ``sm_120a``) that are forbidden as deploy ``arch_id`` values.
    """
    fam = _load_json(_BUILD_FAMILIES_PATH)["compute_capability_to_build_family"]
    key = f"{int(cc[0])}.{int(cc[1])}"
    if key not in fam:
        raise ArchProfileError(f"no build-arch family for compute capability {key}")
    return list(fam[key])


if __name__ == "__main__":  # tiny smoke CLI
    import sys

    if len(sys.argv) > 1:
        prof = load_profile(sys.argv[1])
        print(json.dumps(prof.raw, indent=2))
    else:
        for arch in list_profiles():
            p = load_profile(arch)
            print(f"{p.arch_id:8s} sku={p.sku:16s} sm_count={p.sm_count} "
                  f"cache_layers={p.cache_layers} ldg_e={p.bare_ldg_e_default}")
