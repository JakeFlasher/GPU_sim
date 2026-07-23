"""Typed registry of the cache/memory NCU metric allowlist.

This declares, in one place, the cache/memory metrics the broader
counter-free prediction work targets, each with its measurement geometry:
the NCU token, the value ``kind`` (``fraction`` | ``count`` | ``ratio``),
the unit, the admissible bounds, the ratio denominator/numerator (for
ratios), the cache layers it can describe, the across-repeat
repeatability statistic, the predictor side that estimates it, and the
comparison formula its error geometry uses.

It is intentionally a DECLARATION + VALIDATOR only. It does NOT parse NCU
output and does NOT touch ``tools/local_labels/parse_ncu_metrics.py``: the
percentage-only parser keeps its narrow ``[0,1]``-fraction contract
untouched. A separate typed extraction path that consumes this registry
(for ``count`` and ``ratio`` metrics) and the lockstep wiring into the
driver / comparison contract land later; this module is the shared
declaration both will read.

The allowlist is scoped to cache/memory observables. Occupancy /
launch-occupancy limits, compute throughput (cycles / IPC / MIPS /
warp-instruction counts), and the NCU ``gpu__time_duration`` time metric
are explicitly OUT of scope (wall-clock duration is predicted from the
non-root CUPTI dual-source label, never NCU time). The validator rejects
any entry whose token names one of those forbidden families and any entry
whose declared ``kind`` does not match its comparison geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Pinned registry version. A consumer that joins a typed label against the
# committed contract checks this so a metric-version drift surfaces as a
# typed reason rather than a silent mismatch.
REGISTRY_VERSION = "cache_memory_registry_v1"

# Admissible value kinds.
KIND_FRACTION = "fraction"
KIND_COUNT = "count"
KIND_RATIO = "ratio"
ADMISSIBLE_KINDS = frozenset({KIND_FRACTION, KIND_COUNT, KIND_RATIO})

# Comparison error-geometry tag bound to each kind. A fraction uses an
# epsilon/floor relative error in [0,1]; a count uses an absolute /
# count-floor geometry (never an unbounded relative error); a ratio uses
# a denominator-aware relative error.
KIND_TO_COMPARISON_FORMULA = {
    KIND_FRACTION: "fraction_epsilon_floor",
    KIND_COUNT: "count_absolute_floor",
    KIND_RATIO: "ratio_denominator_relative",
}

# Cache layers a metric may describe.
ADMISSIBLE_LAYERS = frozenset({"L1TEX", "L2", "DRAM"})

# Forbidden token families (substring match, case-insensitive). These mark
# occupancy / launch-limit, compute throughput, and NCU time metrics that
# are out of the cache/memory scope.
_FORBIDDEN_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    ("gpu__time", "NCU time metric (wall-clock comes from the CUPTI dual-source label)"),
    ("occupancy", "occupancy metric"),
    ("launch__", "launch-occupancy-limit metric"),
    ("sm__maximum_warps", "occupancy / launch-limit metric"),
    ("__cycles", "compute-cycle metric"),
    ("sm__cycles", "compute-cycle metric"),
    ("inst_executed", "warp-instruction metric"),
    ("inst_issued", "warp-instruction metric"),
    ("__ipc", "compute IPC metric"),
    ("mio_inst", "compute instruction metric"),
    ("mips", "compute-throughput metric"),
    ("warps_active", "occupancy metric"),
)


class RegistryValidationError(ValueError):
    """Raised when the registry violates its own typing invariants."""


@dataclass(frozen=True)
class MetricSpec:
    """One cache/memory metric and its measurement geometry."""

    key: str
    kind: str
    unit: str
    bounds_lower: float
    bounds_upper: float | None
    admissible_layers: tuple[str, ...]
    repeatability_stat: str
    prediction_source: str
    comparison_formula: str
    ncu_token: str | None = None
    numerator_token: str | None = None
    denominator_token: str | None = None
    derived_from: tuple[str, ...] = field(default_factory=tuple)


def _fraction(key: str, ncu_token: str, layer: str, prediction_source: str) -> MetricSpec:
    return MetricSpec(
        key=key,
        kind=KIND_FRACTION,
        unit="fraction",
        bounds_lower=0.0,
        bounds_upper=1.0,
        admissible_layers=(layer,),
        repeatability_stat="stddev_fraction",
        prediction_source=prediction_source,
        comparison_formula=KIND_TO_COMPARISON_FORMULA[KIND_FRACTION],
        ncu_token=ncu_token,
    )


def _count(key: str, ncu_token: str, unit: str, layer: str) -> MetricSpec:
    return MetricSpec(
        key=key,
        kind=KIND_COUNT,
        unit=unit,
        bounds_lower=0.0,
        bounds_upper=None,
        admissible_layers=(layer,),
        repeatability_stat="count_robust_cv",
        prediction_source="nvbit_trace_count",
        comparison_formula=KIND_TO_COMPARISON_FORMULA[KIND_COUNT],
        ncu_token=ncu_token,
    )


def _derived_count(key: str, unit: str, layer: str, components: tuple[str, ...]) -> MetricSpec:
    return MetricSpec(
        key=key,
        kind=KIND_COUNT,
        unit=unit,
        bounds_lower=0.0,
        bounds_upper=None,
        admissible_layers=(layer,),
        repeatability_stat="count_robust_cv",
        prediction_source="derived_count_total",
        comparison_formula=KIND_TO_COMPARISON_FORMULA[KIND_COUNT],
        derived_from=components,
    )


def _ratio(key: str, numerator: str, denominator: str, layer: str) -> MetricSpec:
    return MetricSpec(
        key=key,
        kind=KIND_RATIO,
        unit="sectors_per_request",
        bounds_lower=0.0,
        bounds_upper=None,
        admissible_layers=(layer,),
        repeatability_stat="ratio_robust_cv",
        prediction_source="derived_sectors_per_request",
        comparison_formula=KIND_TO_COMPARISON_FORMULA[KIND_RATIO],
        numerator_token=numerator,
        denominator_token=denominator,
    )


METRIC_SPECS: tuple[MetricSpec, ...] = (
    # --- Hit rates (fraction) ---
    _fraction(
        "l1tex_sector_hit_rate",
        "l1tex__t_sector_hit_rate.pct",
        "L1TEX",
        "l1_filtered_replay_sector_hit_rate",
    ),
    _fraction(
        "l1tex_global_op_ld_sector_hit_rate",
        "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
        "L1TEX",
        "l1_filtered_replay_sector_hit_rate",
    ),
    _fraction(
        "lts_sector_hit_rate",
        "lts__t_sector_hit_rate.pct",
        "L2",
        "l2_replay_sector_hit_rate",
    ),
    # --- Request counts (count) ---
    _count("l1tex_global_op_ld_requests", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "requests", "L1TEX"),
    _count("l1tex_global_op_st_requests", "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "requests", "L1TEX"),
    _count("l1tex_global_op_atom_requests", "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum", "requests", "L1TEX"),
    _count("l1tex_global_op_red_requests", "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum", "requests", "L1TEX"),
    _count("l1tex_total_requests", "l1tex__t_requests.sum", "requests", "L1TEX"),
    _count("lts_srcunit_tex_requests", "lts__t_requests_srcunit_tex.sum", "requests", "L2"),
    # --- Sector counts (count) ---
    _count("l1tex_global_op_ld_sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "sectors", "L1TEX"),
    _count("l1tex_global_op_st_sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "sectors", "L1TEX"),
    _count("l2_read_sectors_lg_op_ld", "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum", "sectors", "L2"),
    _count("l2_write_sectors_lg_op_st", "l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum", "sectors", "L2"),
    _count("dram_read_sectors", "dram__sectors_read.sum", "sectors", "DRAM"),
    _count("dram_write_sectors", "dram__sectors_write.sum", "sectors", "DRAM"),
    # --- Sectors-per-request ratios (ratio, derived) ---
    _ratio(
        "gmem_ld_sectors_per_request",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "L1TEX",
    ),
    _ratio(
        "gmem_st_sectors_per_request",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
        "L1TEX",
    ),
    # --- Derived totals (count, derived) ---
    _derived_count("gmem_total_requests", "requests", "L1TEX", ("l1tex_global_op_ld_requests", "l1tex_global_op_st_requests")),
    _derived_count("gmem_atom_red_requests", "requests", "L1TEX", ("l1tex_global_op_atom_requests", "l1tex_global_op_red_requests")),
    _derived_count("gmem_total_sectors", "sectors", "L1TEX", ("l1tex_global_op_ld_sectors", "l1tex_global_op_st_sectors")),
    _derived_count("l2_total_sectors", "sectors", "L2", ("l2_read_sectors_lg_op_ld", "l2_write_sectors_lg_op_st")),
    _derived_count("dram_total_sectors", "sectors", "DRAM", ("dram_read_sectors", "dram_write_sectors")),
)


REGISTRY: dict[str, MetricSpec] = {spec.key: spec for spec in METRIC_SPECS}

# All base NCU tokens the registry names (fraction + count entries). The
# later driver / lockstep wiring consumes this; ratios and derived totals
# are computed from these and carry no standalone NCU token.
ALLOWLIST_TOKENS: frozenset[str] = frozenset(
    spec.ncu_token for spec in METRIC_SPECS if spec.ncu_token is not None
)


def forbidden_token_reason(token: str) -> str | None:
    """Return a human-readable reason when ``token`` names a forbidden
    out-of-scope metric family (occupancy / compute / NCU time), else
    ``None``."""
    lowered = token.lower()
    for needle, reason in _FORBIDDEN_SUBSTRINGS:
        if needle in lowered:
            return reason
    return None


def assert_token_admissible(token: str) -> None:
    """Raise ``RegistryValidationError`` when ``token`` is forbidden or is
    not a registered cache/memory base token."""
    reason = forbidden_token_reason(token)
    if reason is not None:
        raise RegistryValidationError(f"token {token!r} is out of scope: {reason}")
    if token not in ALLOWLIST_TOKENS:
        raise RegistryValidationError(
            f"token {token!r} is not in the cache/memory allowlist"
        )


def _validate_spec(spec: MetricSpec) -> list[str]:
    problems: list[str] = []
    if spec.kind not in ADMISSIBLE_KINDS:
        problems.append(f"{spec.key}: kind {spec.kind!r} is not one of {sorted(ADMISSIBLE_KINDS)}")
        return problems

    expected_formula = KIND_TO_COMPARISON_FORMULA[spec.kind]
    if spec.comparison_formula != expected_formula:
        problems.append(
            f"{spec.key}: kind {spec.kind!r} requires comparison_formula "
            f"{expected_formula!r} but declares {spec.comparison_formula!r}"
        )

    for layer in spec.admissible_layers:
        if layer not in ADMISSIBLE_LAYERS:
            problems.append(f"{spec.key}: admissible layer {layer!r} is not one of {sorted(ADMISSIBLE_LAYERS)}")

    if spec.kind == KIND_FRACTION:
        if (spec.bounds_lower, spec.bounds_upper) != (0.0, 1.0):
            problems.append(f"{spec.key}: a fraction must be bounded to [0.0, 1.0]")
        if spec.ncu_token is None:
            problems.append(f"{spec.key}: a fraction must carry an ncu_token")
        if spec.numerator_token is not None or spec.denominator_token is not None:
            problems.append(f"{spec.key}: a fraction must not carry ratio numerator/denominator tokens")
    elif spec.kind == KIND_COUNT:
        if spec.bounds_lower != 0.0 or spec.bounds_upper is not None:
            problems.append(f"{spec.key}: a count must be non-negative and unbounded above")
        if spec.numerator_token is not None or spec.denominator_token is not None:
            problems.append(f"{spec.key}: a count must not carry ratio numerator/denominator tokens")
        base = spec.ncu_token is not None
        derived = bool(spec.derived_from)
        if base == derived:
            problems.append(f"{spec.key}: a count must be either a base ncu_token OR a derived_from total, not both/neither")
        for component in spec.derived_from:
            if component not in REGISTRY:
                problems.append(f"{spec.key}: derived component {component!r} is not in the registry")
    elif spec.kind == KIND_RATIO:
        if spec.bounds_lower != 0.0 or spec.bounds_upper is not None:
            problems.append(f"{spec.key}: a ratio must be non-negative and unbounded above")
        if not spec.numerator_token or not spec.denominator_token:
            problems.append(f"{spec.key}: a ratio must carry both a numerator_token and a denominator_token")
        else:
            for token, role in ((spec.numerator_token, "numerator"), (spec.denominator_token, "denominator")):
                if token not in ALLOWLIST_TOKENS:
                    problems.append(
                        f"{spec.key}: ratio {role}_token {token!r} is not a registered base count token"
                    )
        if spec.ncu_token is not None:
            problems.append(f"{spec.key}: a derived ratio must not carry a standalone ncu_token")

    for token in (spec.ncu_token, spec.numerator_token, spec.denominator_token):
        if token is None:
            continue
        reason = forbidden_token_reason(token)
        if reason is not None:
            problems.append(f"{spec.key}: token {token!r} is out of scope: {reason}")
    return problems


def validate_registry(specs: tuple[MetricSpec, ...] = METRIC_SPECS) -> list[str]:
    """Return a list of registry-typing violations; an empty list means
    the registry is internally consistent."""
    problems: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.key in seen:
            problems.append(f"duplicate registry key {spec.key!r}")
        seen.add(spec.key)
        problems.extend(_validate_spec(spec))
    return problems


def validate_registry_or_raise(specs: tuple[MetricSpec, ...] = METRIC_SPECS) -> None:
    problems = validate_registry(specs)
    if problems:
        raise RegistryValidationError("; ".join(problems))
