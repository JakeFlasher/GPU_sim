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
* A device-tier **SKU** (``rtx_3090_ti``, ``rtx_5060_laptop``,
  ``rtx_5090``) disambiguates
  hardware facts that differ between two cards sharing a compute capability
  (a bare ``sm_120`` cannot tell a GB206 laptop from a GB202 workstation — the
  ``sm_120-laptop`` and ``rtx_5090`` aliases exist precisely to
  prevent that conflation).
* ``build_arch_family`` (a compile-time concern) is kept SEPARATE from the
  deploy/collection ``arch_id`` (Q5): a fatbin may legitimately carry server
  slices ``sm_90a`` / ``sm_120a`` that are never valid deploy arches.

The loader intentionally performs **zero** behaviour change for sm_86: every
sm_86 card field equals the value hardcoded in the live source today, and
``tests/test_arch_profile_lockstep.py`` asserts that byte-for-byte. The historical 5060-laptop card carries the measured 26-SM value; the dedicated
RTX 5090 card carries the vendor-documented 170-SM value. Neither card is a
fallback for the other.

Fields whose real value was never captured on the relevant Blackwell part
(SM120 cache geometry, NVML anchor, bare-LDG default) are represented as an
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
    def runtime_limits(self) -> dict[str, Any] | ResearchGap:
        """Observed launch/identifier envelope, distinct from issue/cache hypotheses."""
        return _gap_or(self.raw.get("runtime_limits"))

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

    def _require_profile_binding(self) -> None:
        """Reject directly constructed contexts that detach a card from its identity."""
        raw = self.profile.raw
        if (
            self.profile.arch_id != self.arch_id
            or raw.get("arch_id") != self.arch_id
            or self.profile.sku != self.sku
            or raw.get("sku") != self.sku
        ):
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: context/profile identity mismatch"
            )
        if resolve_sku(self.sku, strict=True) != self.sku:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: context SKU is not canonical"
            )
        try:
            _card_path_for(self.arch_id, self.sku)
        except ArchProfileError as error:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: architecture/SKU pair is not admitted"
            ) from error

        capability = raw.get("compute_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in capability)
        ):
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: profile compute_capability has invalid shape"
            )
        if resolve_arch_id((capability[0], capability[1])) != self.arch_id:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: profile compute_capability belongs to another architecture"
            )

        card_family = raw.get("build_arch_family")
        families = _load_json(_BUILD_FAMILIES_PATH)[
            "compute_capability_to_build_family"
        ]
        capability_key = f"{capability[0]}.{capability[1]}"
        architecture_family = families.get(capability_key)
        if (
            not isinstance(card_family, list)
            or self.build_arch not in card_family
            or architecture_family is None
            or self.build_arch not in architecture_family
        ):
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: build {self.build_arch!r} is not bound "
                "to the profile architecture and card family"
            )

    def require_issue_resources(self) -> dict[str, dict[str, Any]]:
        """Resolve the complete carded issue-resource scaffold without defaults.

        Production cards currently omit these blocks. Synthetic tests may exercise
        the strict shape, but absence or any whole/nested research gap remains a
        typed failure and never falls back to another architecture.
        """
        self._require_profile_binding()

        def closed_gap_reason(capability: str, value: dict[str, Any]) -> str:
            keys = set(value)
            if (
                not {"status", "reason"} <= keys <= {
                    "status",
                    "reason",
                    "challenge",
                }
                or not isinstance(value.get("reason"), str)
                or (
                    "challenge" in value
                    and not isinstance(value.get("challenge"), str)
                )
            ):
                raise ArchProfileError(
                    f"{self.arch_id}/{self.sku}: capability {capability!r} "
                    "has malformed research_gap shape"
                )
            return value["reason"]

        def require_closed_block(block: str) -> Any:
            value = self.profile.get(block)
            if value is None:
                raise ArchProfileError(
                    f"{self.arch_id}/{self.sku}: capability {block!r} "
                    "unavailable: field absent"
                )
            if is_research_gap(value):
                reason = closed_gap_reason(block, value)
                raise ArchProfileError(
                    f"{self.arch_id}/{self.sku}: capability {block!r} "
                    f"unavailable: {reason}"
                )
            return value

        numeric_shapes = {
            "smsp": (("count", "issue_ports"), (1 << 32) - 1),
            "occupancy": (
                ("max_resident_warps_per_sm", "max_blocks"),
                (1 << 32) - 1,
            ),
            "dep_barriers": (("count",), (1 << 32) - 1),
            # Keep JSON integer transport exact across Python and Rust. Values
            # above 2^53-1 must use a future lossless representation.
            "barrier_latencies": (("read", "non_memory_write"), (1 << 53) - 1),
        }
        resolved: dict[str, dict[str, Any]] = {}
        for block, (fields, maximum) in numeric_shapes.items():
            value = require_closed_block(block)
            if not isinstance(value, dict) or set(value) != set(fields):
                raise ArchProfileError(
                    f"{self.arch_id}/{self.sku}: capability {block!r} has invalid shape"
                )
            normalized: dict[str, int] = {}
            for field in fields:
                scalar = value[field]
                if (
                    isinstance(scalar, bool)
                    or not isinstance(scalar, (int, float))
                    or (isinstance(scalar, float) and not scalar.is_integer())
                    or not 1 <= scalar <= maximum
                ):
                    raise ArchProfileError(
                        f"{self.arch_id}/{self.sku}: capability "
                        f"{block}.{field} must be a positive bounded integer"
                    )
                normalized[field] = int(scalar)
            resolved[block] = normalized

        dual_issue = require_closed_block("dual_issue")
        if not isinstance(dual_issue, dict) or set(dual_issue) != {"supported"}:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: capability 'dual_issue' has invalid shape"
            )
        supported = dual_issue["supported"]
        if is_research_gap(supported):
            reason = closed_gap_reason("dual_issue.supported", supported)
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: capability "
                f"'dual_issue.supported' unavailable: {reason}"
            )
        if not isinstance(supported, bool):
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: capability "
                "dual_issue.supported must be boolean"
            )
        resolved["dual_issue"] = {"supported": supported}
        return resolved

    def require_runtime_limits(self) -> dict[str, Any]:
        """Resolve the card's structural CUDA launch envelope.

        This is intentionally independent of ``require_issue_resources`` and
        never supplies scheduler, cache, or timing defaults. The duplicated
        occupancy block is checked when present so Python and Rust consumers
        cannot observe conflicting resident limits.
        """
        self._require_profile_binding()
        value = self.require("runtime_limits")
        fields = {
            "max_resident_warps_per_sm",
            "max_blocks_per_sm",
            "max_threads_per_sm",
            "max_threads_per_block",
            "sm_id_upper_bound",
            "warp_id_upper_bound",
            "max_shared_memory_per_sm_bytes",
            "max_shared_memory_per_block_bytes",
            "max_shared_memory_per_block_optin_bytes",
            "shared_memory_carveout_kib",
            "provenance",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: runtime_limits has invalid shape"
            )
        positive = (
            "max_resident_warps_per_sm",
            "max_blocks_per_sm",
            "max_threads_per_sm",
            "max_threads_per_block",
            "sm_id_upper_bound",
            "warp_id_upper_bound",
            "max_shared_memory_per_sm_bytes",
            "max_shared_memory_per_block_bytes",
            "max_shared_memory_per_block_optin_bytes",
        )
        normalized: dict[str, Any] = {}
        for field_name in positive:
            raw = value[field_name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or int(raw) != raw or raw <= 0:
                raise ArchProfileError(
                    f"{self.arch_id}/{self.sku}: runtime_limits.{field_name} must be positive integer"
                )
            normalized[field_name] = int(raw)
        carveouts = value["shared_memory_carveout_kib"]
        if (
            not isinstance(carveouts, list)
            or not carveouts
            or any(isinstance(kib, bool) or not isinstance(kib, int) or kib < 0 for kib in carveouts)
            or any(left >= right for left, right in zip(carveouts, carveouts[1:]))
            or any(kib * 1024 > normalized["max_shared_memory_per_sm_bytes"] for kib in carveouts)
        ):
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: runtime_limits.shared_memory_carveout_kib is invalid"
            )
        if normalized["max_threads_per_block"] > normalized["max_threads_per_sm"]:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: per-block thread limit exceeds per-SM limit"
            )
        if normalized["max_threads_per_sm"] < normalized["max_resident_warps_per_sm"] * 32:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: max_threads_per_sm is below resident warp product"
            )
        if not (
            normalized["max_shared_memory_per_block_bytes"]
            <= normalized["max_shared_memory_per_block_optin_bytes"]
            <= normalized["max_shared_memory_per_sm_bytes"]
        ):
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: shared-memory limits are not monotonic"
            )
        sm_count = self.profile.sm_count
        if not isinstance(sm_count, int) or normalized["sm_id_upper_bound"] < sm_count:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: sm identifier bound is below card SM count"
            )
        if normalized["warp_id_upper_bound"] < normalized["max_resident_warps_per_sm"]:
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: warp identifier bound is below resident warp bound"
            )
        if not isinstance(value["provenance"], str) or not value["provenance"].strip():
            raise ArchProfileError(
                f"{self.arch_id}/{self.sku}: runtime_limits.provenance is empty"
            )
        occupancy = self.profile.get("occupancy")
        if occupancy is not None and not is_research_gap(occupancy):
            if not isinstance(occupancy, dict) or occupancy.get("max_resident_warps_per_sm") != normalized["max_resident_warps_per_sm"] or occupancy.get("max_blocks") != normalized["max_blocks_per_sm"]:
                raise ArchProfileError(
                    f"{self.arch_id}/{self.sku}: runtime_limits and occupancy limits diverge"
                )
        normalized["shared_memory_carveout_kib"] = list(carveouts)
        normalized["provenance"] = value["provenance"]
        return normalized


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def list_profiles() -> list[str]:
    """Return the sorted ``arch_id`` values of all installed cards."""
    return sorted({_load_json(p)["arch_id"] for p in _CARDS_DIR.glob("*.json")})


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
    A bare ``sm_120`` selects the active RTX 5090 card; the historical laptop
    card requires ``sm_120-laptop`` or its explicit SKU.
    """
    arch_id = resolve_arch_id(token)
    if sku is None:
        sku = resolve_sku(token, strict=False)
        # The active bare deployment token for CC 12.0 is the cloud RTX 5090
        # target. The historical laptop card remains available only through
        # its explicit SKU/device alias, so a bare sm_120 cannot silently pick
        # the old 5060 facts.
        if sku is None and str(token).strip().lower() == "sm_120":
            sku = "rtx_5090"
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
        arch_cc = next(
            (key for key, value in _aliases()["compute_capability_to_arch_id"].items() if value == arch),
            None,
        )
        family = families.get(arch_cc, [])
        if raw not in family:
            raise ArchProfileError(
                f"build token {raw!r} is not in {arch} family {family!r}"
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
    (``sm_120-laptop`` or an explicit RTX 5090 device name), build tokens
    (``sm_120a``, ``sm_120f``), a
    ``(major, minor)`` compute-capability tuple, or a device-name substring
    ("NVIDIA GeForce RTX 5060 Laptop GPU"). Uncaptured server/workstation
    SKUs raise :class:`ArchProfileError`; the explicitly carded RTX 5090
    Workstation Edition resolves to sm_120.
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
    """Resolve a device-tier SKU (3090 Ti, 5060 laptop, or RTX 5090 WS).

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
    token_sku = {
        "sm_86": "rtx_3090_ti",
        "sm_120-laptop": "rtx_5060_laptop",
    }.get(low)
    if token_sku is not None:
        return token_sku
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
        for card_path in sorted(_CARDS_DIR.glob("*.json")):
            card = _load_json(card_path)
            p = load_profile(card["arch_id"], sku=card.get("sku"))
            print(f"{p.arch_id:8s} sku={p.sku:24s} sm_count={p.sm_count} "
                  f"cache_layers={p.cache_layers} ldg_e={p.bare_ldg_e_default}")
