"""Cache-geometry resolver and validator for the Tier-2 cache
predictor.

Resolves per-launch cache geometry from three sources in the
pinned precedence order:

  1. `cudaDeviceProp::l2CacheSize` (CUDA runtime; primary).
  2. Per-arch static device card descriptor (vendor-documented
     geometry that CUDA does not expose at runtime).
  3. Constant defaults that hold across every current NVIDIA
     arch (sector size, line size).

When neither runtime nor static card declares a capacity-class
field (L2 capacity, num slices, L1 KB, L2 associativity), the
resolver emits `geometry_source="typed_unavailable"` and the
downstream bundle surfaces `predictor_typed_unavailable:
cache_geometry_unresolved`.

A separate validator gates caller-declared geometry against the
live CUDA reading: a declared `l2_capacity_bytes` that disagrees
with the runtime value is rejected with a typed reason.

The resolver/validator split mirrors the cross-study sibling
`/home/jakeshea/3090ti_5060laptop_cross_study/tools/hrd_profiler/
cache_model_spec.py:80-282` - the shape is lifted with the
typed-reason vocabulary renamed to the solbench `cache_geometry_*`
namespace and the field set adapted to the cache-geometry surface
(L1 / L2 / sector / line / slice / associativity /
`geometry_source`).
"""

from __future__ import annotations

from typing import Any


# Sector and line sizes are 32 B / 128 B across every NVIDIA arch
# Ampere through Blackwell — these are safe constant defaults when
# neither CUDA nor a static card declares them.
DEFAULT_SECTOR_SIZE_BYTES: int = 32
DEFAULT_LINE_SIZE_BYTES: int = 128
DEFAULT_SECTORS_PER_LINE: int = (
    DEFAULT_LINE_SIZE_BYTES // DEFAULT_SECTOR_SIZE_BYTES
)


# Per-arch static device card. Values are vendor-documented
# geometry that the CUDA runtime does not expose; fields the
# runtime DOES expose (e.g. ``l2_capacity_bytes``) are duplicated
# here as documented fallbacks for offline / pre-init contexts
# where the runtime reading is absent. Each populated field is
# annotated with a provenance entry in
# ``_STATIC_DEVICE_CARD_PROVENANCE``.
#
# sm_86 (RTX 3090 Ti GA102) is the historical Ampere host — kept
# as a legacy compatibility entry for the landed evidence trail
# under ``.harness_traces/``. sm_120-laptop (RTX 5060 Laptop GB206) is
# retained historical data; the active sm_120 card is the RTX PRO 6000
# Workstation and intentionally has typed gaps below. Undocumented fields are explicitly
# ``None`` rather than guessed; ``resolve_cache_geometry`` routes
# those to per-field ``typed_unavailable`` instead of falling
# through to another arch's defaults.
_STATIC_DEVICE_CARDS: dict[str, dict[str, Any]] = {
    "sm_86": {
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "l1_kb_per_sm": 192,
        "l2_capacity_bytes": 6 * 1024 * 1024,  # 6 MiB
        "num_l2_slices": 6,
        "sector_size_bytes": 32,
        "line_size_bytes": 128,
        "associativity_l2": 16,
        # Constant-memory loads on Ampere sm_86 reach L2
        # architecturally but the calibration label's L2 read
        # denominator (LSU-pipe source-unit class) does NOT count
        # constant-cache traffic. Disabled by default; sm_86 matches
        # the historical behavior byte-for-byte.
        "l2_constant_memory_counts_for_read_score": False,
    },
    "sm_120-laptop": {
        "arch": "sm_120",
        "sku": "rtx_5060_laptop",
        # L1 + shared unified cache: NVIDIA Blackwell consumer
        # documentation describes CC 12.0 with a 128 KB unified
        # L1/texture/shared cache per SM and a maximum runtime-
        # selectable shared capacity of 100 KB
        # (carveouts 0/8/16/32/64/100). The predictor's L1
        # field uses the unified-cache upper bound.
        "l1_kb_per_sm": 128,
        # GB206 L2 capacity (32 MiB) corroborated by the live
        # CUDA runtime probe (cudaDeviceProp.l2CacheSize ==
        # 33554432 on the RTX 5060 Laptop GPU). Provided here
        # for offline contexts; the resolver still prefers the
        # runtime reading.
        "l2_capacity_bytes": 32 * 1024 * 1024,  # 32 MiB
        # Number of L2 slices on GB206: NCU device-attribute probe
        # on the live RTX 5060 Laptop returned
        # device__attribute_l2s_count = 24
        # (corroborated by device__attribute_fbp_count = 2 and
        # device__attribute_num_l2s_per_fbp = 12; 2 * 12 = 24).
        "num_l2_slices": 24,
        # Sector and line sizes are 32 B / 128 B across every
        # NVIDIA arch Ampere through Blackwell, including
        # consumer Blackwell.
        "sector_size_bytes": 32,
        "line_size_bytes": 128,
        # L2 associativity for GB206 is not documented by NVIDIA
        # publicly and NCU does not expose it as a device attribute.
        # an earlier sweep ran an empirical sweep
        # (sm120_l2_associativity_probe/sweep_associativity.py) over
        # candidates {4, 8, 16, 32, 64} against 33 historical-label
        # cells with clean Tier-2 traces. L2 prediction MAPE
        # plateaus at associativity_l2 = 32 (12.8% friendly L2);
        # higher candidates (64) yielded identical numbers,
        # indicating saturation. 32-way is consistent with
        # consumer-Blackwell L2 architecture (Hopper datacenter
        # L2 is 32-way per slice).
        "associativity_l2": 32,
        # Constant-memory L2 read-score denominator flag.
        # DISABLED for sm_120 tonight, despite Blackwell-consumer's
        # texture-source-unit counter including constant-cache
        # routed traffic.
        #
        # Extension A initially flipped this to True; the
        # land revealed that routing constant-memory accesses to
        # the regular L2 LRU simulator produces ~98% hit rates
        # because all constant loads alias to a tiny working set
        # in the LRU model. NCU's actual hit rate for the same
        # cells is 8-76%, reflecting the parallel constant cache's
        # ~8 KiB capacity + broadcast semantics that the regular
        # L2 simulator cannot model. The errors per cell vary
        # markedly (0.22 / 0.29 / 0.53pp), so the isotonic
        # calibrator cannot recover them via a monotonic
        # correction.
        #
        # Disabled here pending Extension A2: a dedicated
        # ConstantCacheSimulator (per-SM, ~8 KiB capacity,
        # fully-associative, empirically calibrated against the
        # 3 revealed cells + holdout). The plumbing through
        # cache_access_policy.route_access /
        # cache_predictor._replay_records_through_caches /
        # cache_geometry.resolve_cache_geometry stays in place
        # so A2 can be flipped on without further plumbing
        # changes.
        "l2_constant_memory_counts_for_read_score": False,
    },
    "sm_120": {
        "arch": "sm_120",
        "sku": "rtx_pro_6000_ws",
        # Pro 6000 cache geometry is intentionally typed unavailable until
        # the cloud host supplies a live device/NCU probe. Never reuse the
        # historical GB206 laptop values above.
        "l1_kb_per_sm": None,
        "l2_capacity_bytes": None,
        "num_l2_slices": None,
        "sector_size_bytes": None,
        "line_size_bytes": None,
        "associativity_l2": None,
        "l2_constant_memory_counts_for_read_score": None,
    },
    # sm_100a (B200 GB100) — populate when B200 access is
    # available (256 KB L1/SM with carveout, 126 MB L2 in 2 die-
    # partitions ~63 MB each per the Blackwell tuning guide).
}


# Per-field provenance for ``_STATIC_DEVICE_CARDS``. Every
# populated field MUST carry a provenance entry naming the
# documentation source or live probe that justifies the value.
# Fields explicitly set to ``None`` in the card do NOT need a
# provenance row — they route to ``typed_unavailable``. The pinning
# test asserts coverage (every non-``None`` field has a provenance
# string).
_STATIC_DEVICE_CARD_PROVENANCE: dict[str, dict[str, str]] = {
    "sm_86": {
        "l1_kb_per_sm": (
            "NVIDIA Ampere GA102 whitepaper: 192 KB unified L1/shared "
            "per SM."
        ),
        "l2_capacity_bytes": (
            "NVIDIA Ampere GA102 whitepaper: 6 MiB L2 cache; "
            "corroborated by cudaDeviceProp.l2CacheSize on RTX 3090 Ti."
        ),
        "num_l2_slices": (
            "NVIDIA Ampere GA102 whitepaper: 6 L2 partitions on GA102."
        ),
        "sector_size_bytes": (
            "NVIDIA Ampere whitepaper: 32 B sector; constant across "
            "Ampere through Blackwell."
        ),
        "line_size_bytes": (
            "NVIDIA Ampere whitepaper: 128 B line; constant across "
            "Ampere through Blackwell."
        ),
        "associativity_l2": (
            "NVIDIA Ampere whitepaper: 16-way L2 slice associativity."
        ),
        "l2_constant_memory_counts_for_read_score": (
            "Ampere sm_86 LSU-pipe source-unit counter "
            "(lts__t_sectors_srcunit_tex_op_read on Ampere) does NOT "
            "include constant-cache-routed traffic; the constant cache "
            "is a separate source-unit class on Ampere. Routing keeps "
            "constant-memory loads architecturally L2-bound but "
            "EXCLUDES them from the L2 read-score denominator so "
            "the simulator's class matches the label's class."
        ),
    },
    "sm_120-laptop": {
        "l1_kb_per_sm": (
            "NVIDIA Blackwell-consumer compute-capability spec (CC "
            "12.0): 128 KB unified L1/texture/shared cache per SM with "
            "configurable shared carveouts up to 100 KB. See "
            "references/NVIDIA_Manuals/cuda_guide/03-02-06_configuring-"
            "l1-shared-memory-balance.md."
        ),
        "l2_capacity_bytes": (
            "Live cudaDeviceProp.l2CacheSize probe on the RTX 5060 "
            "Laptop GPU (NVIDIA Blackwell consumer GB206) returned "
            "33554432 bytes = 32 MiB; corroborated by NVIDIA Blackwell-"
            "consumer documentation for GB206."
        ),
        "num_l2_slices": (
            "Live NCU device-attribute probe on the RTX 5060 Laptop "
            "GPU (NVIDIA Blackwell consumer GB206): "
            "`device__attribute_l2s_count = 24`. Corroborated by "
            "`device__attribute_fbp_count = 2` and "
            "`device__attribute_num_l2s_per_fbp = 12` (2 * 12 = 24). "
            "Captured 2026-05-26 via "
            "`ncu --metrics device__attribute_l2s_count,device__"
            "attribute_fbp_count,device__attribute_num_l2s_per_fbp "
            "<sm_120-binary>` on the live host. Resolves DEC-B "
            "`num_l2_slices` (Tier-1 source per plan)."
        ),
        "sector_size_bytes": (
            "NVIDIA Blackwell tuning guide: 32 B sector; constant "
            "across Ampere through Blackwell."
        ),
        "line_size_bytes": (
            "NVIDIA Blackwell tuning guide: 128 B line; constant "
            "across Ampere through Blackwell."
        ),
        "associativity_l2": (
            "Empirical sweep on 2026-05-26 over candidates "
            "{4, 8, 16, 32, 64} via "
            "`tools/sol_baselines_harness/calibration/cache_prediction/"
            "sm120_l2_associativity_probe/sweep_associativity.py`. "
            "Drove the set-associative simulator on 33 sm_120 "
            "historical-label cells with clean Tier-2 NVBit traces; "
            "fit an isotonic calibrator on the 26 calibration_only "
            "cells and evaluated friendly-cohort L2 MAPE on the 7 "
            "validation_only cells. L2 prediction MAPE plateaus at "
            "associativity_l2 = 32 (friendly L2 = 0.1282 in fraction "
            "domain); the assoc=64 candidate yields identical "
            "numbers, indicating saturation. 32-way is consistent "
            "with NVIDIA's Hopper datacenter L2 (32-way per slice). "
            "This resolves DEC-B's second variable empirically; per "
            "the plan's DEC-B Tradeoff Summary, the sweep was "
            "operator-authorized as an empirical model-selection "
            "step (joel.baker concurrence, 2026-05-26), NOT an "
            "estimate-from-sm_86. See the design notes for details."
        ),
        "l2_constant_memory_counts_for_read_score": (
            "DISABLED (False) for sm_120 pending Extension A2: a "
            "dedicated constant-cache submodel. Extension A "
            "landed the geometry flag + route_access kwarg + "
            "predictor threading, then revealed that routing "
            "constant-memory accesses to the regular L2 LRU simulator "
            "produces ~98% hit rates (LRU alias collapse) while NCU's "
            "actual constant-cache hit rates for the same cells span "
            "8-76% with per-cell variance no monotonic isotonic fit "
            "can recover. Flag stays False until an A2 model lands."
        ),
    },
}


# Producer-side typed reasons emitted by the resolver and
# validator. The full predictor typed-unavailable vocabulary lives
# under the `_typed_unavailable_bundle()` factory; this subset is
# the cache-geometry slice. The full reasons array a caller
# surfaces in a typed-unavailable bundle may include multiple of
# these (the bundle contract requires multi-failure capture, not
# first-failure).
CACHE_GEOMETRY_TYPED_REASONS: frozenset[str] = frozenset({
    "cache_geometry_unresolved",
    "cache_geometry_capacity_missing",
    "cache_geometry_capacity_inconsistent",
    "cache_geometry_capacity_static_card_disagreement",
    "cache_geometry_slices_static_card_disagreement",
})


class CacheGeometryError(Exception):
    """Raised by `validate_declared_cache_geometry` when a
    caller-provided declared geometry contradicts the live CUDA
    reading. Carries the typed-reason string for downstream typed-
    unavailable bundle composition.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


def _resolve_capacity_field(
    *,
    cuda_value: int | None,
    static_card_value: int | None,
    disagreement_note_kind: str,
) -> tuple[int | None, str, list[dict[str, Any]]]:
    """CUDA wins when present and non-zero; static-card fallback
    otherwise. CUDA-vs-static disagreement attaches an info-level
    evidence record (NOT a failure — both values are recorded so
    operators can investigate the divergence). Returns
    `(resolved_value, source_label, evidence_list)` where
    `source_label` is one of `cuda_device_prop`,
    `static_device_card`, or `typed_unavailable`.

    `cuda_value=0` is treated as absent — CUDA sometimes reports
    zero on virtual or pre-init contexts where the device handle
    has not been resolved.
    """
    cuda_present = cuda_value is not None and cuda_value > 0
    static_present = static_card_value is not None and static_card_value > 0
    evidence: list[dict[str, Any]] = []
    if cuda_present:
        assert cuda_value is not None
        if static_present and int(static_card_value) != int(cuda_value):
            evidence.append({
                "result_kind": "info",
                "typed_reason": disagreement_note_kind,
                "cuda_value": int(cuda_value),
                "static_card_value": int(static_card_value),
            })
        return int(cuda_value), "cuda_device_prop", evidence
    if static_present:
        assert static_card_value is not None
        return int(static_card_value), "static_device_card", evidence
    return None, "typed_unavailable", evidence


def resolve_cache_geometry(
    *,
    arch: str,
    cuda_l2_cache_size_bytes: int | None,
    cuda_num_l2_slices: int | None = None,
    static_card: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve the cache-geometry record for the given arch.

    Precedence order:
      cudaDeviceProp > static device card > constant default.

    Inputs:
      arch: SM identifier string (e.g. `"sm_86"`). When `static_card`
        is omitted, the resolver looks up `_STATIC_DEVICE_CARDS[arch]`.
      cuda_l2_cache_size_bytes: live CUDA reading; `None` or `0`
        treated as absent.
      cuda_num_l2_slices: optional live reading (CUDA does not
        actually expose this through cudaDeviceProp today, so most
        callers will pass `None`; left as a parameter for future
        runtime sources that do expose it).
      static_card: optional override; defaults to the registered
        static card for `arch`. Pass `static_card={}` to force the
        no-static-card path.

    Returns `(spec, evidence)`:
      spec: dict carrying every cache-geometry field plus `arch` and
        per-field source labels.
      evidence: list of info-level records (CUDA vs static-card
        disagreements). Empty when both agree or only one source
        is present.

    Field-level sourcing:
      `l2_capacity_bytes`: cudaDeviceProp > static card.
      `num_l2_slices`: cuda_num_l2_slices > static card.
      `l1_kb_per_sm`, `associativity_l2`: static card only (CUDA
        does not expose these directly via cudaDeviceProp).
      `sector_size_bytes`, `line_size_bytes`: static card >
        constant default (32 B / 128 B respectively).

    The aggregate `geometry_source` enum reflects the L2 capacity
    source — the most variable field and the one Codex's required
    test plan explicitly enumerates.
    """
    if static_card is None:
        static_card = _STATIC_DEVICE_CARDS.get(arch, {})

    capacity_bytes, capacity_source, capacity_evidence = (
        _resolve_capacity_field(
            cuda_value=cuda_l2_cache_size_bytes,
            static_card_value=static_card.get("l2_capacity_bytes"),
            disagreement_note_kind=(
                "cache_geometry_capacity_static_card_disagreement"
            ),
        )
    )
    num_slices, num_slices_source, slices_evidence = (
        _resolve_capacity_field(
            cuda_value=cuda_num_l2_slices,
            static_card_value=static_card.get("num_l2_slices"),
            disagreement_note_kind=(
                "cache_geometry_slices_static_card_disagreement"
            ),
        )
    )

    static_l1_kb = static_card.get("l1_kb_per_sm")
    static_associativity_l2 = static_card.get("associativity_l2")
    static_sector_size = static_card.get("sector_size_bytes")
    static_line_size = static_card.get("line_size_bytes")
    static_constant_l2_score = static_card.get(
        "l2_constant_memory_counts_for_read_score"
    )

    sector_size_bytes = (
        int(static_sector_size) if static_sector_size is not None
        else DEFAULT_SECTOR_SIZE_BYTES
    )
    sector_size_source = (
        "static_device_card" if static_sector_size is not None
        else "constant_default"
    )
    line_size_bytes = (
        int(static_line_size) if static_line_size is not None
        else DEFAULT_LINE_SIZE_BYTES
    )
    line_size_source = (
        "static_device_card" if static_line_size is not None
        else "constant_default"
    )

    spec: dict[str, Any] = {
        "arch": arch,
        "l1_kb_per_sm": (
            int(static_l1_kb) if static_l1_kb is not None else None
        ),
        "l1_kb_per_sm_source": (
            "static_device_card" if static_l1_kb is not None
            else "typed_unavailable"
        ),
        "l2_capacity_bytes": capacity_bytes,
        "capacity_source": capacity_source,
        "num_l2_slices": num_slices,
        "num_l2_slices_source": num_slices_source,
        "sector_size_bytes": sector_size_bytes,
        "sector_size_source": sector_size_source,
        "line_size_bytes": line_size_bytes,
        "line_size_source": line_size_source,
        "associativity_l2": (
            int(static_associativity_l2)
            if static_associativity_l2 is not None else None
        ),
        "associativity_l2_source": (
            "static_device_card"
            if static_associativity_l2 is not None
            else "typed_unavailable"
        ),
        # Constant-memory L2 read-score denominator flag. The arch's
        # static device card opts in (True) when its
        # texture-source-unit counter on the L2 includes
        # constant-cache-routed traffic (Blackwell-consumer behavior);
        # otherwise the flag defaults to False (Ampere LSU-pipe
        # semantics). Resolver defaults to False if the card does
        # not specify the field, preserving sm_86 byte-equality
        # for any under-specified arch.
        "l2_constant_memory_counts_for_read_score": bool(
            static_constant_l2_score
        ) if static_constant_l2_score is not None else False,
        "l2_constant_memory_counts_for_read_score_source": (
            "static_device_card"
            if static_constant_l2_score is not None
            else "constant_default"
        ),
        # The aggregate `geometry_source` enum mirrors the L2
        # capacity source. The surface contract carries a single
        # `geometry_source` enum but the per-field labels above
        # let a caller report partial resolutions honestly.
        "geometry_source": capacity_source,
    }

    evidence: list[dict[str, Any]] = []
    evidence.extend(capacity_evidence)
    evidence.extend(slices_evidence)
    return spec, evidence


def validate_declared_cache_geometry(
    declared: dict[str, Any],
    *,
    cuda_l2_cache_size_bytes: int | None,
) -> None:
    """Validate a caller-declared `cache_geometry` block against
    the live CUDA `l2CacheSize`. Used by callers that ship a
    pre-computed geometry record (e.g., a calibration sidecar
    that was generated on a different host) to confirm the
    declared capacity matches the runtime.

    Typed reasons:
      `cache_geometry_capacity_missing`: declared `l2_capacity_bytes`
        is absent/zero AND CUDA's `l2CacheSize` is absent/zero.
      `cache_geometry_capacity_inconsistent`: declared
        `l2_capacity_bytes` is positive AND CUDA's `l2CacheSize`
        is positive AND they disagree. CUDA wins by design in the
        resolver, but the validator rejects callers that ship a
        record contradicting the runtime — this surfaces
        host-divergence early instead of silently overwriting.

    The corroboration arm (CUDA vs static-card disagreement
    during resolution) is NOT this validator's responsibility —
    that lives in the resolver's
    `cache_geometry_capacity_static_card_disagreement` info
    evidence.
    """
    declared_capacity = declared.get("l2_capacity_bytes")
    declared_present = (
        declared_capacity is not None and declared_capacity > 0
    )
    cuda_present = (
        cuda_l2_cache_size_bytes is not None
        and cuda_l2_cache_size_bytes > 0
    )
    if not declared_present and not cuda_present:
        raise CacheGeometryError(
            "cache_geometry_capacity_missing",
            f"declared l2_capacity_bytes={declared_capacity} and "
            f"CUDA l2CacheSize={cuda_l2_cache_size_bytes}; both "
            f"absent",
        )
    if (
        declared_present
        and cuda_present
        and int(declared_capacity) != int(cuda_l2_cache_size_bytes)
    ):
        raise CacheGeometryError(
            "cache_geometry_capacity_inconsistent",
            f"declared l2_capacity_bytes={declared_capacity} "
            f"contradicts CUDA l2CacheSize="
            f"{cuda_l2_cache_size_bytes}",
        )


__all__ = [
    "CacheGeometryError",
    "CACHE_GEOMETRY_TYPED_REASONS",
    "DEFAULT_SECTOR_SIZE_BYTES",
    "DEFAULT_LINE_SIZE_BYTES",
    "DEFAULT_SECTORS_PER_LINE",
    "resolve_cache_geometry",
    "validate_declared_cache_geometry",
]
