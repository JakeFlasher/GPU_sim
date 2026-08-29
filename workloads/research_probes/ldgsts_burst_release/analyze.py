#!/usr/bin/env python3
"""Offline schema and frozen reduction for the LDGSTS burst assay.

This module is intentionally local-only. It freezes the 40-cell manifest,
validates one process's paired raw records, retains achieved delays per gap
edge, and applies the frozen process/cell decision rules. It remains pure:
capture-tree discovery, NCU admission, and live execution are supplied by a
caller as already-admitted reductions rather than performed here.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


BURSTS = 16_384
WARMUP = 64
TIMING_SCHEMA = 2
DIRECT_REPETITIONS = 7
NCU_REPEATS_MIN = 3
NCU_REPEATS_MAX = 5

IDENTIFICATION_DEPTHS = (1, 4, 5, 6, 8)
KNEE_DEPTHS = (4, 5, 6)
SPACING_GAPS = (16, 64)
ROUTES = ("l1", "l2", "dram")

ROLE_COUNTS = {
    "identification": 10,
    "spacing": 12,
    "holdout": 5,
    "placement": 9,
    "clock_assay": 4,
}

PAIR_FIELDS = frozenset(
    {
        "type",
        "timing_schema",
        "cell",
        "burst",
        "warp",
        "arm_order",
        "issue_on",
        "issue_off",
        "drain_on",
        "drain_off",
        "gaps_on",
        "gaps_off",
        "smid_on_before",
        "smid_on_after",
        "smid_off_before",
        "smid_off_after",
        "error_mask",
        "correct",
    }
)


class AnalysisError(RuntimeError):
    """A malformed or inadmissible input."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object, field: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field}: expected a number",
    )
    result = float(value)
    require(math.isfinite(result), f"{field}: expected a finite number")
    return result


@dataclass(frozen=True, slots=True)
class CellSpec:
    key: str
    role: str
    assay: str
    warps: int
    route: str
    depth: int
    gap: int


def _cell_key(assay: str, warps: int, route: str, depth: int, gap: int) -> str:
    return f"{assay}_w{warps}_{route}_d{depth}_g{gap}"


def _cell(
    role: str,
    assay: str,
    warps: int,
    route: str,
    depth: int,
    gap: int,
) -> CellSpec:
    return CellSpec(
        _cell_key(assay, warps, route, depth, gap),
        role,
        assay,
        warps,
        route,
        depth,
        gap,
    )


def expected_cells() -> tuple[CellSpec, ...]:
    """Return the exact 40-cell partition in deterministic role order."""

    cells: list[CellSpec] = []
    for route in ("l1", "dram"):
        for depth in IDENTIFICATION_DEPTHS:
            cells.append(_cell("identification", "ldgsts", 1, route, depth, 0))
    for route in ("l1", "dram"):
        for depth in KNEE_DEPTHS:
            for gap in SPACING_GAPS:
                cells.append(_cell("spacing", "ldgsts", 1, route, depth, gap))
    for depth in IDENTIFICATION_DEPTHS:
        cells.append(_cell("holdout", "ldgsts", 1, "l2", depth, 0))
    for route in ROUTES:
        for depth in KNEE_DEPTHS:
            cells.append(_cell("placement", "ldgsts", 4, route, depth, 0))
    for depth in (4, 5, 6, 8):
        cells.append(_cell("clock_assay", "ldg", 1, "l1", depth, 0))

    require(len(cells) == 40, "internal matrix did not produce 40 cells")
    require(len({cell.key for cell in cells}) == 40, "internal matrix has duplicate keys")
    return tuple(cells)


EXPECTED_CELLS = expected_cells()
EXPECTED_BY_KEY = {cell.key: cell for cell in EXPECTED_CELLS}


def validate_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    """Validate the exact keys, fields, and frozen campaign cardinalities."""

    require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
        "matrix cells must be a sequence",
    )
    require(len(rows) == 40, "matrix must contain exactly 40 cells")
    actual: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"matrix cell {index}: expected an object")
        key = row.get("key")
        require(isinstance(key, str) and key, f"matrix cell {index}: invalid key")
        require(key not in actual, f"matrix: duplicate key {key}")
        actual[key] = row

    require(set(actual) == set(EXPECTED_BY_KEY), "matrix keys differ from the frozen partition")
    for key, expected in EXPECTED_BY_KEY.items():
        row = actual[key]
        for field in ("role", "assay", "route"):
            require(
                isinstance(row.get(field), str),
                f"{key}: {field} must be a string",
            )
        for field in ("warps", "depth", "gap"):
            require(
                _is_int(row.get(field)),
                f"{key}: {field} must be an integer",
            )
        for field in ("role", "assay", "warps", "route", "depth", "gap"):
            require(
                row.get(field) == getattr(expected, field),
                f"{key}: {field} differs from the frozen matrix",
            )
        require(
            _is_int(row.get("timing_schema"))
            and row.get("timing_schema") == TIMING_SCHEMA,
            f"{key}: unsupported timing schema",
        )
        for field_name, expected_value in (
            ("bursts", BURSTS),
            ("warmup", WARMUP),
            ("direct_repetitions", DIRECT_REPETITIONS),
            ("ncu_repeats_min", NCU_REPEATS_MIN),
            ("ncu_repeats_max", NCU_REPEATS_MAX),
        ):
            require(
                _is_int(row.get(field_name))
                and row.get(field_name) == expected_value,
                f"{key}: invalid {field_name}",
            )
        require(
            row.get("arms") == ["enabled", "off"],
            f"{key}: wrong paired arms",
        )

    partition = {
        role: tuple(cell.key for cell in EXPECTED_CELLS if cell.role == role)
        for role in ROLE_COUNTS
    }
    require(
        {role: len(keys) for role, keys in partition.items()} == ROLE_COUNTS,
        "matrix role partition does not match 10/12/5/9/4",
    )
    return partition


@dataclass(frozen=True, slots=True)
class ProcessReduction:
    """One process repetition; W4 slots and gap edges remain separate."""

    cell: str
    repetition: int
    smid: int
    issue_by_slot: tuple[float, ...]
    completion_by_slot: tuple[float, ...]
    gap_on_by_slot: tuple[tuple[float, ...], ...]
    gap_off_by_slot: tuple[tuple[float, ...], ...]
    arm_order_pattern: tuple[str, ...] = ()


def _gap_bounds(gap: int) -> tuple[float, float]:
    if gap == 0:
        return (0.0, 0.0)
    if gap == 16:
        return (16.0, 38.0)
    if gap == 64:
        return (64.0, 96.0)
    raise AnalysisError(f"unsupported gap sentinel {gap}")


def d_i(row: Mapping[str, Any]) -> float:
    """Raw issue effect for one paired burst."""

    return _finite_number(row["issue_on"], "issue_on") - _finite_number(
        row["issue_off"], "issue_off"
    )


def d_c(row: Mapping[str, Any]) -> float:
    """Raw post-issue completion effect for one paired burst."""

    return (
        _finite_number(row["drain_on"], "drain_on")
        - _finite_number(row["issue_on"], "issue_on")
    ) - (
        _finite_number(row["drain_off"], "drain_off")
        - _finite_number(row["issue_off"], "issue_off")
    )


def reduce_process_rows(
    rows: Sequence[Mapping[str, Any]],
    spec: CellSpec,
    repetition: int,
    *,
    expected_bursts: int = BURSTS,
) -> ProcessReduction:
    """Validate and reduce one process without pooling bursts or W4 slots."""

    require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
        "paired rows must be a sequence",
    )
    require(_is_int(repetition) and repetition >= 1, "invalid process repetition")
    require(_is_int(expected_bursts) and expected_bursts > 0, "invalid burst count")
    slots = tuple(range(spec.warps))
    expected_pairs = {
        (burst, slot) for burst in range(expected_bursts) for slot in slots
    }
    observed: dict[tuple[int, int], Mapping[str, Any]] = {}
    burst_orders: dict[int, str] = {}
    smids: set[int] = set()
    gap_count = max(spec.depth - 1, 0)
    gap_low, gap_high = _gap_bounds(spec.gap)

    for row_index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"row {row_index}: expected an object")
        require(set(row) == PAIR_FIELDS, f"row {row_index}: paired schema mismatch")
        require(row["type"] == "paired", f"row {row_index}: expected type=paired")
        require(
            _is_int(row["timing_schema"]) and row["timing_schema"] == TIMING_SCHEMA,
            f"row {row_index}: unsupported timing schema",
        )
        require(row["cell"] == spec.key, f"row {row_index}: wrong cell")
        burst, slot = row["burst"], row["warp"]
        require(_is_int(burst), f"row {row_index}: burst must be an integer")
        require(_is_int(slot), f"row {row_index}: warp must be an integer")
        pair = (burst, slot)
        require(pair in expected_pairs, f"row {row_index}: unexpected burst/warp pair")
        require(pair not in observed, f"row {row_index}: duplicate burst/warp pair")
        observed[pair] = row

        order = row["arm_order"]
        require(order in {"on_off", "off_on"}, f"row {row_index}: invalid arm order")
        require(
            burst_orders.setdefault(burst, order) == order,
            f"burst {burst}: W4 slots disagree on arm order",
        )
        require(row["correct"] is True, f"row {row_index}: correctness failed")
        require(
            _is_int(row["error_mask"]) and row["error_mask"] == 0,
            f"row {row_index}: nonzero or malformed error mask",
        )

        row_smids: list[int] = []
        for field in (
            "smid_on_before",
            "smid_on_after",
            "smid_off_before",
            "smid_off_after",
        ):
            value = row[field]
            require(_is_int(value) and value >= 0, f"row {row_index}: invalid {field}")
            row_smids.append(value)
        require(len(set(row_smids)) == 1, f"row {row_index}: SM changed inside pair")
        smids.add(row_smids[0])

        for field in ("issue_on", "issue_off", "drain_on", "drain_off"):
            require(
                _finite_number(row[field], f"row {row_index} {field}") >= 0,
                f"row {row_index}: {field} must be nonnegative",
            )
        require(
            _finite_number(row["drain_on"], "drain_on")
            >= _finite_number(row["issue_on"], "issue_on"),
            f"row {row_index}: on drain endpoint precedes issue endpoint",
        )
        require(
            _finite_number(row["drain_off"], "drain_off")
            >= _finite_number(row["issue_off"], "issue_off"),
            f"row {row_index}: off drain endpoint precedes issue endpoint",
        )
        for field in ("gaps_on", "gaps_off"):
            gaps = row[field]
            require(isinstance(gaps, list), f"row {row_index}: {field} must be a list")
            require(len(gaps) == gap_count, f"row {row_index}: wrong {field} edge count")
            for edge, value in enumerate(gaps):
                delay = _finite_number(value, f"row {row_index} {field}[{edge}]")
                require(
                    gap_low <= delay <= gap_high,
                    f"row {row_index}: {field}[{edge}] outside sentinel range",
                )

    require(set(observed) == expected_pairs, "process has missing burst/warp pairs")
    require(len(smids) == 1, "cell migrated between SMs within one process")
    order_counts = {
        order: sum(value == order for value in burst_orders.values())
        for order in ("on_off", "off_on")
    }
    require(
        abs(order_counts["on_off"] - order_counts["off_on"]) <= 1,
        "enabled/off arm order is not balanced across bursts",
    )

    issue_by_slot: list[float] = []
    completion_by_slot: list[float] = []
    gap_on_by_slot: list[tuple[float, ...]] = []
    gap_off_by_slot: list[tuple[float, ...]] = []
    for slot in slots:
        slot_rows = [observed[(burst, slot)] for burst in range(expected_bursts)]
        issue_by_slot.append(
            float(
                statistics.median(d_i(row) for row in slot_rows)
            )
        )
        # The target records drain_* as an endpoint duration from the same arm
        # start as issue_*.  Form the post-issue completion effect explicitly,
        # so the raw reduction remains identical if endpoint fields are later
        # retained at a higher precision.
        completion_by_slot.append(
            float(
                statistics.median(
                    d_c(row) for row in slot_rows
                )
            )
        )
        for field, destination in (
            ("gaps_on", gap_on_by_slot),
            ("gaps_off", gap_off_by_slot),
        ):
            destination.append(
                tuple(
                    float(
                        statistics.median(
                            _finite_number(row[field][edge], f"{field}[{edge}]")
                            for row in slot_rows
                        )
                    )
                    for edge in range(gap_count)
                )
            )

    return ProcessReduction(
        spec.key,
        repetition,
        next(iter(smids)),
        tuple(issue_by_slot),
        tuple(completion_by_slot),
        tuple(gap_on_by_slot),
        tuple(gap_off_by_slot),
        tuple(burst_orders[burst] for burst in range(expected_bursts)),
    )


@dataclass(frozen=True, slots=True)
class EffectSeries:
    """Seven process-level values and their unscaled MAD."""

    values: tuple[float, ...]
    median: float
    mad: float


@dataclass(frozen=True, slots=True)
class ContrastSeries:
    """A named seven-repetition contrast and its median/MAD summary."""

    name: str
    values: tuple[float, ...]
    median: float
    mad: float


Series = EffectSeries | ContrastSeries


def _named_series(name: str, values: Iterable[float]) -> ContrastSeries:
    summary = summarize_processes(values)
    return ContrastSeries(name, summary.values, summary.median, summary.mad)


def _process_index(
    reductions: Mapping[str, Sequence[ProcessReduction]],
    key: str,
) -> dict[int, ProcessReduction]:
    """Validate one cell's seven process reductions and index by repetition."""

    require(isinstance(reductions, Mapping), "reductions must be a mapping")
    require(key in reductions, f"missing process reductions for {key}")
    processes = reductions[key]
    require(
        isinstance(processes, Sequence) and not isinstance(processes, (str, bytes)),
        f"{key}: process reductions must be a sequence",
    )
    require(
        len(processes) == DIRECT_REPETITIONS,
        f"{key}: expected seven process reductions",
    )
    indexed: dict[int, ProcessReduction] = {}
    expected = EXPECTED_BY_KEY.get(key)
    require(expected is not None, f"unknown frozen cell {key}")
    for process in processes:
        require(
            isinstance(process, ProcessReduction),
            f"{key}: malformed process reduction",
        )
        require(process.cell == key, f"{key}: process has wrong cell")
        require(
            _is_int(process.repetition)
            and 1 <= process.repetition <= DIRECT_REPETITIONS,
            f"{key}: repetition must be in 1..7",
        )
        require(
            process.repetition not in indexed,
            f"{key}: duplicate repetition {process.repetition}",
        )
        require(_is_int(process.smid) and process.smid >= 0, f"{key}: invalid SMID")
        require(
            isinstance(process.arm_order_pattern, tuple)
            and all(order in {"on_off", "off_on"} for order in process.arm_order_pattern),
            f"{key}: invalid retained arm-order pattern",
        )
        counts = {
            order: process.arm_order_pattern.count(order)
            for order in ("on_off", "off_on")
        }
        require(
            len(process.arm_order_pattern) >= 2
            and counts["on_off"] > 0
            and counts["off_on"] > 0,
            f"{key}: retained arm-order pattern is incomplete",
        )
        require(
            abs(counts["on_off"] - counts["off_on"]) <= 1,
            f"{key}: retained arm-order pattern is unbalanced",
        )
        for field_name in ("issue_by_slot", "completion_by_slot"):
            values = getattr(process, field_name)
            require(
                isinstance(values, tuple) and len(values) == expected.warps,
                f"{key}: {field_name} slot cardinality mismatch",
            )
            for value in values:
                _finite_number(value, f"{key}:{field_name}")
        for field_name in ("gap_on_by_slot", "gap_off_by_slot"):
            values = getattr(process, field_name)
            require(
                isinstance(values, tuple) and len(values) == expected.warps,
                f"{key}: {field_name} slot cardinality mismatch",
            )
            for edges in values:
                require(
                    isinstance(edges, tuple) and len(edges) == expected.depth - 1,
                    f"{key}: {field_name} edge cardinality mismatch",
                )
                for value in edges:
                    _finite_number(value, f"{key}:{field_name}")
        indexed[process.repetition] = process
    require(
        set(indexed) == set(range(1, DIRECT_REPETITIONS + 1)),
        f"{key}: repetitions must be exactly 1 through 7",
    )
    return indexed


def cell_series(
    reductions: Mapping[str, Sequence[ProcessReduction]],
    key: str,
    *,
    field_name: str = "issue_by_slot",
    slot: int = 0,
    name: str | None = None,
) -> ContrastSeries:
    """Summarize one retained field by process without pooling W4 slots."""

    require(field_name in {"issue_by_slot", "completion_by_slot"}, "unsupported effect field")
    expected = EXPECTED_BY_KEY.get(key)
    require(expected is not None, f"unknown frozen cell {key}")
    require(_is_int(slot) and 0 <= slot < expected.warps, f"{key}: invalid slot")
    indexed = _process_index(reductions, key)
    values = [
        _finite_number(getattr(indexed[repetition], field_name)[slot], field_name)
        for repetition in range(1, DIRECT_REPETITIONS + 1)
    ]
    return _named_series(name or f"{field_name}:{key}:s{slot}", values)


def contrast_series(
    reductions: Mapping[str, Sequence[ProcessReduction]],
    left_key: str,
    right_key: str,
    *,
    field_name: str = "issue_by_slot",
    slot: int = 0,
    name: str | None = None,
) -> ContrastSeries:
    """Form a matched-repetition, same-SM cell contrast."""

    require(left_key != right_key, "contrast requires two distinct cells")
    left = _process_index(reductions, left_key)
    right = _process_index(reductions, right_key)
    left_spec = EXPECTED_BY_KEY.get(left_key)
    right_spec = EXPECTED_BY_KEY.get(right_key)
    require(left_spec is not None and right_spec is not None, "unknown frozen contrast cell")
    require(field_name in {"issue_by_slot", "completion_by_slot"}, "unsupported effect field")
    require(
        left_spec.warps == right_spec.warps,
        f"contrast slot cardinality differs: {left_key} vs {right_key}",
    )
    require(_is_int(slot) and 0 <= slot < left_spec.warps, "invalid contrast slot")
    values: list[float] = []
    for repetition in range(1, DIRECT_REPETITIONS + 1):
        left_pattern = left[repetition].arm_order_pattern
        right_pattern = right[repetition].arm_order_pattern
        require(
            left_pattern and right_pattern and left_pattern == right_pattern,
            f"contrast arm-order schedules are missing or differ at repetition {repetition}",
        )
        require(
            left[repetition].smid == right[repetition].smid,
            f"contrast crosses SMs at repetition {repetition}: {left_key} vs {right_key}",
        )
        lhs = _finite_number(getattr(left[repetition], field_name)[slot], field_name)
        rhs = _finite_number(getattr(right[repetition], field_name)[slot], field_name)
        values.append(lhs - rhs)
    return _named_series(name or f"{field_name}:{left_key}-{right_key}:s{slot}", values)


def difference_series(left: Series, right: Series, *, name: str) -> ContrastSeries:
    """Subtract two already matched seven-value series element by element."""

    require(_well_formed_series(left), "left effect series is malformed")
    require(_well_formed_series(right), "right effect series is malformed")
    require(len(left.values) == DIRECT_REPETITIONS, "left effect must have seven values")
    require(len(right.values) == DIRECT_REPETITIONS, "right effect must have seven values")
    return _named_series(name, (lhs - rhs for lhs, rhs in zip(left.values, right.values)))


def knee_key(route: str, gap: int) -> str:
    return f"{route}:g{gap}"


def route_key(depth: int, gap: int) -> str:
    return f"n{depth}:g{gap}"


@dataclass(frozen=True, slots=True)
class PSKR:
    """Frozen P/S/K/R issue effects plus per-cell E/C series."""

    p: Mapping[str, ContrastSeries]
    s: Mapping[str, ContrastSeries]
    k: Mapping[str, ContrastSeries]
    r: Mapping[str, ContrastSeries]
    e: Mapping[str, ContrastSeries]
    c: Mapping[str, ContrastSeries]
    missing: tuple[str, ...] = ()
    active_warps: int = 1
    slot: int = 0

    def as_mapping(self) -> dict[str, Mapping[str, ContrastSeries]]:
        return {"P": self.p, "S": self.s, "K": self.k, "R": self.r, "E": self.e, "C": self.c}


def reduce_pskr(
    reductions: Mapping[str, Sequence[ProcessReduction]],
    *,
    slot: int = 0,
    active_warps: int | None = None,
) -> PSKR:
    """Compute available frozen P/S/K/R contrasts for one retained slot.

    Slot zero defaults to the strict W1 matrix. Pass ``active_warps=4`` for a
    W4 placement capture; the explicit selector prevents a missing W1 cell
    from being silently replaced by a W4 slot-zero cell.
    """

    require(isinstance(reductions, Mapping), "reductions must be a mapping")
    require(_is_int(slot) and 0 <= slot < 4, "invalid reduction slot")
    require(
        all(isinstance(key, str) and key in EXPECTED_BY_KEY for key in reductions),
        "reductions contain an unknown frozen cell",
    )
    if active_warps is None:
        active_warps = 4 if slot > 0 else 1
    require(
        _is_int(active_warps)
        and active_warps in {1, 4}
        and slot < active_warps,
        "invalid active-warp/slot pair",
    )

    def pick(assay: str, route: str, depth: int, gap: int) -> str | None:
        key = _cell_key(assay, active_warps, route, depth, gap)
        return key if key in reductions else None

    e = {
        key: cell_series(reductions, key, slot=slot)
        for key in reductions
        if key in EXPECTED_BY_KEY
        and EXPECTED_BY_KEY[key].assay == "ldgsts"
        and EXPECTED_BY_KEY[key].warps == active_warps
    }
    c = {
        key: cell_series(reductions, key, field_name="completion_by_slot", slot=slot)
        for key in reductions
        if key in EXPECTED_BY_KEY
        and EXPECTED_BY_KEY[key].assay == "ldgsts"
        and EXPECTED_BY_KEY[key].warps == active_warps
    }
    p: dict[str, ContrastSeries] = {}
    s: dict[str, ContrastSeries] = {}
    k: dict[str, ContrastSeries] = {}
    r: dict[str, ContrastSeries] = {}
    missing: list[str] = []
    for route in ("l1", "dram", "l2"):
        gaps = (0,) if active_warps == 4 or route == "l2" else (0, *SPACING_GAPS)
        for gap in gaps:
            if route == "l2" and gap != 0:
                continue
            d4 = pick("ldgsts", route, 4, gap)
            d5 = pick("ldgsts", route, 5, gap)
            d6 = pick("ldgsts", route, 6, gap)
            if not all((d4, d5, d6)):
                missing.append(f"K:{knee_key(route, gap)}")
                continue
            p_value = contrast_series(reductions, d5, d4, slot=slot, name=f"P:{route}:g{gap}")
            s_value = contrast_series(reductions, d6, d5, slot=slot, name=f"S:{route}:g{gap}")
            p[knee_key(route, gap)] = p_value
            s[knee_key(route, gap)] = s_value
            k[knee_key(route, gap)] = difference_series(
                s_value, p_value, name=f"K:{route}:g{gap}"
            )
    for gap in ((0,) if active_warps == 4 else (0, *SPACING_GAPS)):
        depths = (
            KNEE_DEPTHS
            if active_warps == 4
            else IDENTIFICATION_DEPTHS
            if gap == 0
            else KNEE_DEPTHS
        )
        for depth in depths:
            l1 = pick("ldgsts", "l1", depth, gap)
            dram = pick("ldgsts", "dram", depth, gap)
            if l1 is None or dram is None:
                missing.append(f"R:{route_key(depth, gap)}")
                continue
            r[route_key(depth, gap)] = contrast_series(
                reductions,
                dram,
                l1,
                slot=slot,
                name=f"R:n{depth}:g{gap}",
            )
    return PSKR(
        p,
        s,
        k,
        r,
        e,
        c,
        tuple(sorted(set(missing))),
        active_warps,
        slot,
    )


def reduce_slot_pskr(
    reductions: Mapping[str, Sequence[ProcessReduction]],
    slot: int,
) -> PSKR:
    """Convenience wrapper for one of the four W4 placement slots."""

    return reduce_pskr(reductions, slot=slot, active_warps=4)


def summarize_processes(values: Iterable[float]) -> EffectSeries:
    samples = tuple(_finite_number(value, "process effect") for value in values)
    require(len(samples) == DIRECT_REPETITIONS, "effect must contain seven processes")
    center = float(statistics.median(samples))
    mad = float(statistics.median(abs(value - center) for value in samples))
    return EffectSeries(samples, center, mad)


def _well_formed_series(effect: object) -> bool:
    if not isinstance(effect, (EffectSeries, ContrastSeries)):
        return False
    try:
        if len(effect.values) != DIRECT_REPETITIONS:
            return False
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in effect.values
        ):
            return False
        if not isinstance(effect.median, (int, float)) or isinstance(effect.median, bool):
            return False
        if not isinstance(effect.mad, (int, float)) or isinstance(effect.mad, bool):
            return False
        if not math.isfinite(float(effect.median)) or not math.isfinite(float(effect.mad)):
            return False
        center = float(statistics.median(float(value) for value in effect.values))
        mad = float(
            statistics.median(abs(float(value) - center) for value in effect.values)
        )
        return math.isclose(
            float(effect.median), center, abs_tol=1e-12, rel_tol=0.0
        ) and math.isclose(float(effect.mad), mad, abs_tol=1e-12, rel_tol=0.0)
    except (TypeError, ValueError, statistics.StatisticsError):
        return False


def stable(effect: Series) -> bool:
    return _well_formed_series(effect) and effect.mad <= 0.5


def op_eq(effect: Series) -> bool:
    return stable(effect) and abs(effect.median) <= 1.0


def pos(effect: Series) -> bool:
    return stable(effect) and effect.median > 2.0


def ordered(effect: Series) -> bool:
    return stable(effect) and effect.median > 0.0


def opposite_op_eq(effect: Series) -> bool:
    """Return whether a stable effect is materially outside equality."""

    return stable(effect) and abs(effect.median) > 2.0


def opposite_pos(effect: Series) -> bool:
    """Return whether a stable effect is the declared opposite of ``pos``."""

    return op_eq(effect) or (stable(effect) and effect.median < -2.0)


def opposite_ordered(effect: Series) -> bool:
    """Return whether a stable effect is the declared opposite of ``ordered``."""

    return stable(effect) and effect.median <= 0.0


IDENTIFICATION_LABELS = (
    "route_invariant_relief_with_short_gap",
    "route_invariant_relief_only_with_long_gap",
    "route_sensitive_issue_response",
    "no_material_ldgsts_issue_knee",
)


@dataclass(frozen=True, slots=True)
class StageResult:
    """One staged decision with explicit status and retained predicates."""

    status: str
    failures: tuple[str, ...] = ()
    predicates: Mapping[str, bool] = field(default_factory=dict)
    effects: Mapping[str, Series] = field(default_factory=dict)
    successor_eligible: bool = False
    identification_label: str | None = None


@dataclass(frozen=True, slots=True)
class StagedResult:
    """Identification, placement, and holdout results kept independent."""

    identification: StageResult
    placement: StageResult
    holdout: StageResult
    successor_eligible: bool


def _effect_container(
    effects: PSKR | Mapping[str, Any],
    family: str,
) -> Mapping[str, Series]:
    """Return one family from a PSKR object or a nested/flat mapping."""

    if isinstance(effects, PSKR):
        return getattr(effects, family.lower(), {})
    if not isinstance(effects, Mapping):
        return {}
    for candidate in (family, family.lower(), family.upper()):
        value = effects.get(candidate)
        if isinstance(value, Mapping):
            return value
    return effects


def _lookup_effect(
    effects: PSKR | Mapping[str, Any],
    family: str,
    key: str,
) -> Series | None:
    """Look up a named effect while accepting the documented flat/nested forms."""

    container = _effect_container(effects, family)
    candidates = (key, f"{family}:{key}", f"{family.lower()}:{key}")
    for candidate in candidates:
        value = container.get(candidate) if isinstance(container, Mapping) else None
        if isinstance(value, (EffectSeries, ContrastSeries)):
            return value
    # A flat map may already use a fully qualified key.
    if isinstance(effects, Mapping):
        for candidate in candidates:
            value = effects.get(candidate)
            if isinstance(value, (EffectSeries, ContrastSeries)):
                return value

    # Derive adjacent route effects from retained per-cell E series when a
    # caller passes the PSKR object produced by reduce_pskr().
    if family in {"A", "B"}:
        depth, gap = _parse_depth_gap_key(key)
        if depth is not None and gap is not None:
            route_left, route_right = ("l2", "l1") if family == "A" else ("dram", "l2")
            for active_warps in (1, 4):
                left_key = _cell_key("ldgsts", active_warps, route_left, depth, gap)
                right_key = _cell_key("ldgsts", active_warps, route_right, depth, gap)
                left = _lookup_effect(effects, "E", left_key)
                right = _lookup_effect(effects, "E", right_key)
                if left is not None and right is not None:
                    return difference_series(left, right, name=f"{family}:{key}")
    return None


def _parse_depth_gap_key(key: str) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"(?:n|d)?(\d+)(?::?g(\d+))?", key)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2) or 0)


def _required_effects(
    effects: PSKR | Mapping[str, Any],
    requests: Iterable[tuple[str, str]],
) -> tuple[dict[str, Series], tuple[str, ...]]:
    found: dict[str, Series] = {}
    missing: list[str] = []
    for family, key in requests:
        name = f"{family}:{key}"
        try:
            value = _lookup_effect(effects, family, key)
        except AnalysisError as error:
            missing.append(f"{name} ({error})")
            continue
        if value is None:
            missing.append(name)
        elif not _well_formed_series(value):
            missing.append(f"{name} (malformed series)")
        else:
            found[name] = value
    return found, tuple(missing)


def classify_identification(
    *,
    admission_ok: bool,
    clock_knee: Series | None = None,
    effects: PSKR | Mapping[str, Any],
    admission_failures: Iterable[str] = (),
) -> StageResult:
    """Lock exactly one W1 label, without opening diagnostics.

    ``clock_knee`` is normally supplied from the separate LDG control assay;
    PSKR itself contains only LDGSTS effects and cannot manufacture it.
    """

    failures = tuple(str(item) for item in admission_failures)
    if not admission_ok:
        return StageResult("invalid_experiment", failures or ("identification admission failed",))
    if clock_knee is None:
        clock_knee = _lookup_effect(effects, "LDG_K", "l1:g0")
    if clock_knee is None:
        return StageResult("invalid_clock_assay", ("missing LDG clock-assay knee",))
    if not pos(clock_knee):
        return StageResult(
            "invalid_clock_assay",
            ("LDG clock-assay K is not positive material",),
            {"clock_knee_positive_material": False},
            {"LDG_K:l1:g0": clock_knee},
        )

    requests: list[tuple[str, str]] = []
    for route in ("l1", "dram"):
        for gap in (0, 16, 64):
            requests.append(("K", knee_key(route, gap)))
    for depth in IDENTIFICATION_DEPTHS:
        requests.append(("R", route_key(depth, 0)))
    found, missing = _required_effects(effects, requests)
    found["LDG_K:l1:g0"] = clock_knee
    if missing:
        return StageResult(
            "invalid_experiment",
            tuple(f"missing identification effect {item}" for item in missing),
            {"clock_knee_positive_material": True},
            found,
        )

    def get(family: str, key: str) -> Series:
        value = found[f"{family}:{key}"]
        return value

    predicates: dict[str, bool] = {"clock_knee_positive_material": True}
    knee0 = all(pos(get("K", knee_key(route, 0))) for route in ("l1", "dram"))
    route_eq0 = all(
        op_eq(get("R", route_key(depth, 0))) for depth in IDENTIFICATION_DEPTHS
    )
    predicates["knee0"] = knee0
    predicates["route_eq0"] = route_eq0
    predicates["rule_route_invariant_relief_with_short_gap"] = (
        knee0
        and route_eq0
        and op_eq(get("K", knee_key("l1", 16)))
        and op_eq(get("K", knee_key("dram", 16)))
    )
    predicates["rule_route_invariant_relief_only_with_long_gap"] = (
        knee0
        and route_eq0
        and pos(get("K", knee_key("l1", 16)))
        and pos(get("K", knee_key("dram", 16)))
        and op_eq(get("K", knee_key("l1", 64)))
        and op_eq(get("K", knee_key("dram", 64)))
    )
    predicates["rule_route_sensitive_issue_response"] = (
        knee0
        and all(
            op_eq(get("R", route_key(depth, 0)))
            for depth in (1, 4, 5)
        )
        and all(
            pos(get("R", route_key(depth, 0)))
            for depth in (6, 8)
        )
        and op_eq(get("K", knee_key("l1", 16)))
        and pos(get("K", knee_key("dram", 16)))
    )
    predicates["rule_no_material_ldgsts_issue_knee"] = (
        route_eq0
        and all(
            op_eq(get("K", knee_key(route, gap)))
            for route in ("l1", "dram")
            for gap in (0, 16, 64)
        )
    )
    matches = [
        label
        for label in IDENTIFICATION_LABELS
        if predicates[f"rule_{label}"]
    ]
    status = matches[0] if len(matches) == 1 else "valid_ambiguous_no_model"
    if not matches:
        failures = ("zero identification rules matched",)
    elif len(matches) > 1:
        failures = (f"multiple identification rules matched: {', '.join(matches)}",)
    return StageResult(
        status,
        failures,
        predicates,
        found,
        identification_label=status if status in IDENTIFICATION_LABELS else None,
    )


def _slot_effect(
    effects: Mapping[str, Any] | PSKR,
    family: str,
    key: str,
) -> Series | None:
    return _lookup_effect(effects, family, key)


def _total_route_effect(
    effects: Mapping[str, Any] | PSKR,
    depth: int,
    gap: int,
) -> Series | None:
    direct = _slot_effect(effects, "R", route_key(depth, gap))
    if direct is not None:
        return direct
    left = _slot_effect(effects, "A", route_key(depth, gap))
    right = _slot_effect(effects, "B", route_key(depth, gap))
    if left is None or right is None:
        return None
    return _named_series(
        f"R:{route_key(depth, gap)}",
        (lhs + rhs for lhs, rhs in zip(left.values, right.values)),
    )


def _placement_effects_for_slot(
    effects: Mapping[str, Any] | PSKR,
    *,
    require_total: bool,
) -> tuple[dict[str, Series], tuple[str, ...]]:
    requests: list[tuple[str, str]] = []
    for route in ROUTES:
        requests.append(("K", knee_key(route, 0)))
    for depth in KNEE_DEPTHS:
        requests.extend(
            (("A", route_key(depth, 0)), ("B", route_key(depth, 0)))
        )
    found, missing = _required_effects(effects, requests)
    total = _total_route_effect(effects, 6, 0)
    if total is None:
        if require_total:
            missing = (*missing, f"R:{route_key(6, 0)}")
    else:
        found[f"R:{route_key(6, 0)}"] = total
    return found, missing


def classify_placement(
    *,
    identification: StageResult,
    admission_ok: bool,
    effects_by_slot: Mapping[int, PSKR | Mapping[str, Any]],
    admission_failures: Iterable[str] = (),
    w1_effects: PSKR | Mapping[str, Any] | None = None,
) -> StageResult:
    """Classify W4 as compatible/associated/falsified/inconclusive only."""

    failures = tuple(str(item) for item in admission_failures)
    if not admission_ok:
        return StageResult("invalid", failures or ("placement admission failed",))
    if identification.status not in IDENTIFICATION_LABELS:
        return StageResult("not_applicable", ("identification is not uniquely named",))
    if (
        not isinstance(effects_by_slot, Mapping)
        or set(effects_by_slot) != {0, 1, 2, 3}
        or any(not _is_int(slot) for slot in effects_by_slot)
    ):
        return StageResult("invalid", ("placement must retain exactly four warp slots",))
    if isinstance(w1_effects, PSKR) and (
        not _is_int(w1_effects.active_warps)
        or not _is_int(w1_effects.slot)
        or w1_effects.active_warps != 1
        or w1_effects.slot != 0
    ):
        return StageResult("invalid", ("W1 baseline provenance mismatch",))

    slot_results: dict[int, dict[str, bool]] = {}
    all_effects: dict[str, Series] = {}
    missing: list[str] = []
    needs_total = (
        identification.status == "route_sensitive_issue_response"
        or w1_effects is not None
    )
    for slot in range(4):
        try:
            source_effects = effects_by_slot[slot]
            if isinstance(source_effects, PSKR) and (
                not _is_int(source_effects.active_warps)
                or not _is_int(source_effects.slot)
                or source_effects.active_warps != 4
                or source_effects.slot != slot
            ):
                raise AnalysisError(
                    f"slot provenance mismatch: expected W4 slot {slot}, "
                    f"got W{source_effects.active_warps}/s{source_effects.slot}"
                )
            found, absent = _placement_effects_for_slot(
                source_effects,
                require_total=needs_total,
            )
        except AnalysisError as error:
            found, absent = {}, (str(error),)
        if absent:
            missing.extend(f"s{slot}:{item}" for item in absent)
        all_effects.update({f"s{slot}:{key}": value for key, value in found.items()})
        knees = {route: found.get(f"K:{knee_key(route, 0)}") for route in ROUTES}
        a = {depth: found.get(f"A:{route_key(depth, 0)}") for depth in KNEE_DEPTHS}
        b = {depth: found.get(f"B:{route_key(depth, 0)}") for depth in KNEE_DEPTHS}
        total = found.get(f"R:{route_key(6, 0)}")
        required_values_for_presence = (*knees.values(), *a.values(), *b.values())
        if needs_total:
            required_values_for_presence = (*required_values_for_presence, total)
        if any(value is None for value in required_values_for_presence):
            continue
        invariant = all(pos(value) for value in knees.values()) and all(
            op_eq(a[depth]) and op_eq(b[depth]) for depth in KNEE_DEPTHS
        )
        sensitive = (
            all(pos(value) for value in knees.values())
            and all(op_eq(a[depth]) and op_eq(b[depth]) for depth in (4, 5))
            and ordered(a[6])
            and ordered(b[6])
            and pos(total)
        )
        no_knee = all(op_eq(value) for value in knees.values()) and all(
            op_eq(a[depth]) and op_eq(b[depth]) for depth in KNEE_DEPTHS
        )
        if identification.status == "no_material_ldgsts_issue_knee":
            knee_opposite = any(opposite_op_eq(value) for value in knees.values())
            edge_opposite = any(
                opposite_op_eq(a[depth]) or opposite_op_eq(b[depth])
                for depth in KNEE_DEPTHS
            )
            total_opposite = False
        elif identification.status == "route_sensitive_issue_response":
            knee_opposite = any(opposite_pos(value) for value in knees.values())
            edge_opposite = any(
                opposite_op_eq(a[depth]) or opposite_op_eq(b[depth])
                for depth in (4, 5)
            ) or opposite_ordered(a[6]) or opposite_ordered(b[6])
            total_opposite = opposite_pos(total)
        else:
            knee_opposite = any(opposite_pos(value) for value in knees.values())
            edge_opposite = any(
                opposite_op_eq(a[depth]) or opposite_op_eq(b[depth])
                for depth in KNEE_DEPTHS
            )
            total_opposite = False
        if identification.status == "route_sensitive_issue_response":
            required_values: tuple[Series, ...] = (
                *knees.values(),
                a[4],
                a[5],
                a[6],
                b[4],
                b[5],
                b[6],
                total,
            )
        else:
            required_values = (*knees.values(), *a.values(), *b.values())
        slot_results[slot] = {
            "route_invariant_relief": invariant,
            "route_sensitive_issue_response": sensitive,
            "no_material_ldgsts_issue_knee": no_knee,
            "all_required_stable": all(stable(value) for value in required_values),
            "opposite_class": knee_opposite or edge_opposite or total_opposite,
        }

    if missing:
        return StageResult("invalid", tuple(sorted(set(f"missing placement effect {item}" for item in missing))), {}, all_effects)
    expected_rule = {
        "route_invariant_relief_with_short_gap": "route_invariant_relief",
        "route_invariant_relief_only_with_long_gap": "route_invariant_relief",
        "route_sensitive_issue_response": "route_sensitive_issue_response",
        "no_material_ldgsts_issue_knee": "no_material_ldgsts_issue_knee",
    }[identification.status]
    predicates: dict[str, bool] = {}
    for slot, values in slot_results.items():
        for key, value in values.items():
            predicates[f"s{slot}:{key}"] = value
    compatible = all(values.get(expected_rule, False) for values in slot_results.values())
    stable_all = all(values.get("all_required_stable", False) for values in slot_results.values())
    slot_classes = {
        tuple(values.get(label, False) for label in (
            "route_invariant_relief",
            "route_sensitive_issue_response",
            "no_material_ldgsts_issue_knee",
        ))
        for values in slot_results.values()
    }
    slot_disagreement = len(slot_classes) > 1
    material_only_w4 = False
    if w1_effects is not None:
        w1_total = _total_route_effect(w1_effects, 6, 0)
        w4_totals = [
            _total_route_effect(effects_by_slot[slot], 6, 0)
            for slot in range(4)
        ]
        material_only_w4 = (
            w1_total is not None
            and stable(w1_total)
            and not pos(w1_total)
            and any(value is not None and pos(value) for value in w4_totals)
        )
    predicates["slot_disagreement"] = slot_disagreement
    predicates["material_only_w4"] = material_only_w4
    if compatible:
        status = "compatible"
    elif stable_all and (slot_disagreement or material_only_w4):
        status = "placement_associated"
    elif stable_all and any(values.get("opposite_class", False) for values in slot_results.values()):
        status = "falsified"
    else:
        status = "inconclusive"
    return StageResult(
        status,
        (),
        predicates,
        all_effects,
        identification_label=identification.status,
    )


def _holdout_effects(
    effects: PSKR | Mapping[str, Any],
) -> tuple[dict[str, Series], tuple[str, ...]]:
    requests: list[tuple[str, str]] = [("K", "l2:g0")]
    for depth in IDENTIFICATION_DEPTHS:
        requests.extend(
            (("A", route_key(depth, 0)), ("B", route_key(depth, 0)))
        )
    return _required_effects(effects, requests)


def classify_l2(
    *,
    identification: StageResult,
    admission_ok: bool,
    effects: PSKR | Mapping[str, Any],
    w1_total_route: Series | None = None,
    admission_failures: Iterable[str] = (),
) -> StageResult:
    """Classify the unopened L2 route without changing W1 identification."""

    failures = tuple(str(item) for item in admission_failures)
    if not admission_ok:
        return StageResult("invalid", failures or ("L2 admission failed",))
    if identification.status not in IDENTIFICATION_LABELS:
        return StageResult("not_applicable", ("identification is not uniquely named",))
    found, missing = _holdout_effects(effects)
    if w1_total_route is None and isinstance(effects, PSKR):
        w1_total_route = _total_route_effect(effects, 6, 0)
    if w1_total_route is None and identification.status == "route_sensitive_issue_response":
        missing = (*missing, "W1 total DRAM-minus-L1 depth-6 effect")
    if missing:
        return StageResult("invalid", tuple(f"missing holdout effect {item}" for item in missing), {}, found)
    l2_knee = found["K:l2:g0"]
    a = {depth: found[f"A:{route_key(depth, 0)}"] for depth in IDENTIFICATION_DEPTHS}
    b = {depth: found[f"B:{route_key(depth, 0)}"] for depth in IDENTIFICATION_DEPTHS}
    predicates: dict[str, bool] = {}
    predicates["l2_knee_positive_material"] = pos(l2_knee)
    predicates["l2_knee_operationally_equal"] = op_eq(l2_knee)
    for depth in IDENTIFICATION_DEPTHS:
        predicates[f"A_n{depth}_operationally_equal"] = op_eq(a[depth])
        predicates[f"B_n{depth}_operationally_equal"] = op_eq(b[depth])
        predicates[f"A_n{depth}_ordered"] = ordered(a[depth])
        predicates[f"B_n{depth}_ordered"] = ordered(b[depth])
    if w1_total_route is not None:
        found["R:n6:g0"] = w1_total_route
        predicates["w1_total_route_positive_material"] = pos(w1_total_route)
    if identification.status in {
        "route_invariant_relief_with_short_gap",
        "route_invariant_relief_only_with_long_gap",
    }:
        required_values: tuple[Series, ...] = (
            l2_knee,
            *a.values(),
            *b.values(),
        )
        required = predicates["l2_knee_positive_material"] and all(
            predicates[f"A_n{depth}_operationally_equal"]
            and predicates[f"B_n{depth}_operationally_equal"]
            for depth in IDENTIFICATION_DEPTHS
        )
        opposite = opposite_pos(l2_knee) or any(
            opposite_op_eq(a[depth]) or opposite_op_eq(b[depth])
            for depth in IDENTIFICATION_DEPTHS
        )
    elif identification.status == "route_sensitive_issue_response":
        required_values = (
            l2_knee,
            a[1],
            a[4],
            a[5],
            a[6],
            a[8],
            b[1],
            b[4],
            b[5],
            b[6],
            b[8],
            w1_total_route,
        ) if w1_total_route is not None else (
            l2_knee,
            *a.values(),
            *b.values(),
        )
        required = (
            predicates["l2_knee_positive_material"]
            and all(
                predicates[f"A_n{depth}_operationally_equal"]
                and predicates[f"B_n{depth}_operationally_equal"]
                for depth in (1, 4, 5)
            )
            and predicates["A_n6_ordered"]
            and predicates["B_n6_ordered"]
            and predicates["A_n8_ordered"]
            and predicates["B_n8_ordered"]
            and predicates.get("w1_total_route_positive_material", False)
        )
        opposite = (
            opposite_pos(l2_knee)
            or any(
                opposite_op_eq(a[depth]) or opposite_op_eq(b[depth])
                for depth in (1, 4, 5)
            )
            or any(
                opposite_ordered(a[depth]) or opposite_ordered(b[depth])
                for depth in (6, 8)
            )
            or (w1_total_route is not None and opposite_pos(w1_total_route))
        )
    else:
        required_values = (l2_knee, *a.values(), *b.values())
        required = predicates["l2_knee_operationally_equal"] and all(
            predicates[f"A_n{depth}_operationally_equal"]
            and predicates[f"B_n{depth}_operationally_equal"]
            for depth in IDENTIFICATION_DEPTHS
        )
        opposite = opposite_op_eq(l2_knee) or any(
            opposite_op_eq(a[depth]) or opposite_op_eq(b[depth])
            for depth in IDENTIFICATION_DEPTHS
        )
    predicates["all_required_stable"] = all(stable(value) for value in required_values)
    predicates["required_for_confirmation"] = required
    predicates["stable_opposite_class_observed"] = opposite and predicates[
        "all_required_stable"
    ]
    if required:
        status = "confirmed"
    elif predicates["stable_opposite_class_observed"]:
        status = "falsified"
    else:
        status = "inconclusive"
    return StageResult(
        status,
        (),
        predicates,
        found,
        identification_label=identification.status,
    )


def classify_staged(
    *,
    identification: StageResult,
    placement: StageResult,
    holdout: StageResult,
) -> StagedResult:
    """Join stages while preserving the no-relabel rule and eligibility gate."""

    eligible = (
        identification.status in IDENTIFICATION_LABELS
        and identification.identification_label == identification.status
        and placement.status == "compatible"
        and holdout.status == "confirmed"
        and placement.identification_label == identification.status
        and holdout.identification_label == identification.status
    )
    return StagedResult(identification, placement, holdout, eligible)


@dataclass(frozen=True, slots=True)
class GapAdmission:
    admitted: bool
    failures: tuple[str, ...]
    differences: Mapping[str, EffectSeries]


def admit_gap_matching(
    reductions: Mapping[str, Sequence[ProcessReduction]],
) -> GapAdmission:
    """Compare every spacing edge with its same-edge fixed reference.

    For each sentinel, enabled W1/L1/depth-6 supplies all five reference
    edges. Differences remain process-level vectors, and every candidate arm
    and edge must be stable and operationally equal to its reference.
    """

    expected = {cell.key: cell for cell in EXPECTED_CELLS if cell.role == "spacing"}
    if set(reductions) != set(expected):
        missing = sorted(set(expected) - set(reductions))
        extra = sorted(set(reductions) - set(expected))
        return GapAdmission(
            False,
            (f"spacing keys mismatch: missing={missing}, extra={extra}",),
            {},
        )

    failures: list[str] = []
    indexed: dict[str, dict[int, ProcessReduction]] = {}
    for key, spec in expected.items():
        try:
            by_repetition = _process_index(reductions, key)
        except AnalysisError as error:
            failures.append(f"{key}: {error}")
            continue
        low, high = _gap_bounds(spec.gap)
        edge_values = (
            value
            for process in by_repetition.values()
            for field in (process.gap_on_by_slot, process.gap_off_by_slot)
            for edges in field
            for value in edges
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not low <= float(value) <= high
            for value in edge_values
        ):
            failures.append(f"{key}: invalid retained achieved gap")
            continue
        indexed[key] = by_repetition

    if failures:
        return GapAdmission(False, tuple(failures), {})

    differences: dict[str, EffectSeries] = {}
    for gap in SPACING_GAPS:
        reference_key = _cell_key("ldgsts", 1, "l1", 6, gap)
        for key, spec in expected.items():
            if spec.gap != gap:
                continue
            for arm, attribute in (("on", "gap_on_by_slot"), ("off", "gap_off_by_slot")):
                for edge in range(spec.depth - 1):
                    values: list[float] = []
                    for repetition in range(1, DIRECT_REPETITIONS + 1):
                        reference = indexed[reference_key][repetition]
                        candidate = indexed[key][repetition]
                        if reference.smid != candidate.smid:
                            failures.append(
                                f"g{gap}:{key}:{arm}:e{edge}: matched process crosses SMs"
                            )
                        source = getattr(candidate, attribute)[0][edge]
                        # Compare each arm to the corresponding arm in the
                        # fixed reference.  Using the enabled baseline for
                        # the off arm would silently accept/reject spacing
                        # when the two reference schedules differ.
                        baseline = getattr(reference, attribute)[0][edge]
                        values.append(source - baseline)
                    name = f"g{gap}:{key}:{arm}:e{edge}"
                    effect = summarize_processes(values)
                    differences[name] = effect
                    if not op_eq(effect):
                        failures.append(
                            f"{name}: achieved gap differs from reference "
                            f"(median={effect.median}, MAD={effect.mad})"
                        )

    require(len(differences) == 96, "internal gap comparison cardinality is not 96")
    return GapAdmission(not failures, tuple(failures), differences)


def _manifest_fixture() -> list[dict[str, object]]:
    return [
        {
            "key": cell.key,
            "timing_schema": TIMING_SCHEMA,
            "role": cell.role,
            "assay": cell.assay,
            "warps": cell.warps,
            "route": cell.route,
            "depth": cell.depth,
            "gap": cell.gap,
            "bursts": BURSTS,
            "warmup": WARMUP,
            "arms": ["enabled", "off"],
            "direct_repetitions": DIRECT_REPETITIONS,
            "ncu_repeats_min": NCU_REPEATS_MIN,
            "ncu_repeats_max": NCU_REPEATS_MAX,
        }
        for cell in EXPECTED_CELLS
    ]


def _paired_fixture(spec: CellSpec, bursts: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for burst in range(bursts):
        order = "on_off" if burst % 2 == 0 else "off_on"
        for slot in range(spec.warps):
            gaps = [float(spec.gap)] * max(spec.depth - 1, 0)
            rows.append(
                {
                    "type": "paired",
                    "timing_schema": TIMING_SCHEMA,
                    "cell": spec.key,
                    "burst": burst,
                    "warp": slot,
                    "arm_order": order,
                    "issue_on": 10 + burst + slot,
                    "issue_off": 2 + burst,
                    # Cumulative endpoint durations from each arm start.  The
                    # values intentionally make endpoint subtraction differ
                    # from a bare drain-field subtraction.
                    "drain_on": 30 + burst + 2 * slot,
                    "drain_off": 7 + burst,
                    "gaps_on": gaps,
                    "gaps_off": gaps,
                    "smid_on_before": 3,
                    "smid_on_after": 3,
                    "smid_off_before": 3,
                    "smid_off_after": 3,
                    "error_mask": 0,
                    "correct": True,
                }
            )
    return rows


def _spacing_fixture() -> dict[str, list[ProcessReduction]]:
    result: dict[str, list[ProcessReduction]] = {}
    for spec in EXPECTED_CELLS:
        if spec.role != "spacing":
            continue
        result[spec.key] = [
            reduce_process_rows(
                _paired_fixture(spec, 4), spec, repetition, expected_bursts=4
            )
            for repetition in range(1, DIRECT_REPETITIONS + 1)
        ]
    return result


def _process_reduction_fixture(
    spec: CellSpec,
    repetition: int,
    issue: float,
    *,
    completion: float | None = None,
    smid: int = 3,
) -> ProcessReduction:
    """Build one already-reduced process row for offline contrast fixtures."""

    if completion is None:
        completion = issue + 5.0
    gaps = tuple(float(spec.gap) for _ in range(max(spec.depth - 1, 0)))
    return ProcessReduction(
        spec.key,
        repetition,
        smid,
        tuple(issue + slot for slot in range(spec.warps)),
        tuple(completion + slot for slot in range(spec.warps)),
        (gaps,) * spec.warps,
        (gaps,) * spec.warps,
        tuple("on_off" if burst % 2 == 0 else "off_on" for burst in range(4)),
    )


def _w1_pskr_fixture() -> dict[str, list[ProcessReduction]]:
    """Create a complete W1 LDGSTS matrix with known P/S/K/R effects."""

    result: dict[str, list[ProcessReduction]] = {}
    for spec in EXPECTED_CELLS:
        if spec.assay != "ldgsts" or spec.warps != 1:
            continue
        processes: list[ProcessReduction] = []
        for repetition in range(1, DIRECT_REPETITIONS + 1):
            if spec.depth in KNEE_DEPTHS:
                desired_k = 3.0
                base = 100.0
                if spec.depth == 4:
                    issue = base
                elif spec.depth == 5:
                    issue = base + 2.0
                else:
                    issue = base + 2.0 + (2.0 + desired_k)
            else:
                issue = 40.0 + spec.depth
            # A common per-repetition drift cancels inside every matched
            # contrast but remains visible in the seven raw vectors.
            issue += repetition * 0.25
            processes.append(
                _process_reduction_fixture(
                    spec,
                    repetition,
                    issue,
                    completion=issue + 7.0,
                )
            )
        result[spec.key] = processes
    return result


def _constant_series(value: float, name: str) -> ContrastSeries:
    return _named_series(name, (value,) * DIRECT_REPETITIONS)


def _identification_effect_fixture(label: str) -> dict[str, Mapping[str, Series]]:
    effects: dict[str, dict[str, Series]] = {"K": {}, "R": {}}
    for route in ("l1", "dram"):
        for gap in (0, 16, 64):
            value = 0.0
            if gap == 0:
                value = 3.0
            if label == "ambiguous" and gap == 16:
                value = 1.5
            if label == "route_invariant_relief_only_with_long_gap" and gap == 16:
                value = 3.0
            if label == "route_sensitive_issue_response" and gap == 16 and route == "dram":
                value = 3.0
            effects["K"][knee_key(route, gap)] = _constant_series(
                value, f"K:{route}:g{gap}"
            )
    for depth in IDENTIFICATION_DEPTHS:
        value = 3.0 if label == "route_sensitive_issue_response" and depth in (6, 8) else 0.0
        effects["R"][route_key(depth, 0)] = _constant_series(
            value, f"R:n{depth}:g0"
        )
    if label == "no_material_ldgsts_issue_knee":
        for key in effects["K"]:
            effects["K"][key] = _constant_series(0.0, key)
    if label == "route_sensitive_issue_response":
        effects["K"][knee_key("l1", 16)] = _constant_series(0.0, "K:l1:g16")
    return effects


def _route_stage_fixture(
    *,
    knee: Mapping[str, float],
    a: Mapping[int, float],
    b: Mapping[int, float],
    total: float | None = None,
) -> dict[str, Mapping[str, Series]]:
    effects: dict[str, dict[str, Series]] = {"K": {}, "A": {}, "B": {}, "R": {}}
    for route in ROUTES:
        effects["K"][knee_key(route, 0)] = _constant_series(
            knee[route], f"K:{route}:g0"
        )
    for depth in a:
        effects["A"][route_key(depth, 0)] = _constant_series(
            a[depth], f"A:n{depth}:g0"
        )
        effects["B"][route_key(depth, 0)] = _constant_series(
            b[depth], f"B:n{depth}:g0"
        )
    if total is not None:
        effects["R"][route_key(6, 0)] = _constant_series(total, "R:n6:g0")
    return effects


def _l2_stage_fixture(
    *,
    knee: float,
    a: Mapping[int, float],
    b: Mapping[int, float],
    total: float | None = None,
) -> dict[str, Mapping[str, Series]]:
    effects: dict[str, dict[str, Series]] = {
        "K": {"l2:g0": _constant_series(knee, "K:l2:g0")},
        "A": {},
        "B": {},
        "R": {},
    }
    for depth in a:
        effects["A"][route_key(depth, 0)] = _constant_series(
            a[depth], f"A:n{depth}:g0"
        )
        effects["B"][route_key(depth, 0)] = _constant_series(
            b[depth], f"B:n{depth}:g0"
        )
    if total is not None:
        effects["R"][route_key(6, 0)] = _constant_series(total, "R:n6:g0")
    return effects


def _expect_error(call: Callable[[], object]) -> None:
    try:
        call()
    except AnalysisError:
        return
    raise AssertionError("expected AnalysisError")


def self_test() -> None:
    _expect_error(lambda: validate_matrix(None))
    partition = validate_matrix(_manifest_fixture())
    assert {role: len(keys) for role, keys in partition.items()} == ROLE_COUNTS
    duplicate = _manifest_fixture()
    duplicate[-1] = dict(duplicate[0])
    _expect_error(lambda: validate_matrix(duplicate))
    wrong_warmup = _manifest_fixture()
    wrong_warmup[0]["warmup"] = WARMUP + 1
    _expect_error(lambda: validate_matrix(wrong_warmup))
    wrong_gap_type = _manifest_fixture()
    wrong_gap_type[0]["gap"] = False
    _expect_error(lambda: validate_matrix(wrong_gap_type))
    wrong_schema_type = _manifest_fixture()
    wrong_schema_type[0]["timing_schema"] = float(TIMING_SCHEMA)
    _expect_error(lambda: validate_matrix(wrong_schema_type))

    spec = EXPECTED_BY_KEY["ldgsts_w4_l1_d4_g0"]
    rows = _paired_fixture(spec, 4)
    _expect_error(lambda: reduce_process_rows(None, spec, 1, expected_bursts=4))
    reduced = reduce_process_rows(rows, spec, 1, expected_bursts=4)
    assert d_i(rows[0]) == 8.0
    assert d_c(rows[0]) == 15.0
    assert reduced.issue_by_slot == (8.0, 9.0, 10.0, 11.0)
    assert reduced.completion_by_slot == (15.0, 16.0, 17.0, 18.0)
    assert reduced.completion_by_slot[0] != 23.0
    assert reduced.gap_on_by_slot == ((0.0, 0.0, 0.0),) * 4
    _expect_error(
        lambda: reduce_process_rows(
            rows + [dict(rows[0])], spec, 1, expected_bursts=4
        )
    )
    wrong_error_type = [dict(row) for row in rows]
    wrong_error_type[0]["error_mask"] = False
    _expect_error(
        lambda: reduce_process_rows(wrong_error_type, spec, 1, expected_bursts=4)
    )
    wrong_endpoint = [dict(row) for row in rows]
    wrong_endpoint[0]["drain_on"] = wrong_endpoint[0]["issue_on"] - 1
    _expect_error(
        lambda: reduce_process_rows(wrong_endpoint, spec, 1, expected_bursts=4)
    )

    spacing_spec = EXPECTED_BY_KEY["ldgsts_w1_l1_d6_g16"]
    spacing_reduction = reduce_process_rows(
        _paired_fixture(spacing_spec, 4), spacing_spec, 1, expected_bursts=4
    )
    assert spacing_reduction.gap_on_by_slot == ((16.0,) * 5,)
    spacing = _spacing_fixture()
    admission = admit_gap_matching(spacing)
    assert admission.admitted and len(admission.differences) == 96

    # Keep an arm-asymmetric reference to ensure off-arm matching uses the
    # off-arm reference, rather than accidentally comparing against enabled.
    arm_asymmetric = {
        key: [
            ProcessReduction(
                process.cell,
                process.repetition,
                process.smid,
                process.issue_by_slot,
                process.completion_by_slot,
                process.gap_on_by_slot,
                tuple(
                    tuple(value + 3.0 for value in edges)
                    for edges in process.gap_off_by_slot
                ),
                process.arm_order_pattern,
            )
            for process in processes
        ]
        for key, processes in spacing.items()
    }
    asymmetric_admission = admit_gap_matching(arm_asymmetric)
    assert asymmetric_admission.admitted
    assert asymmetric_admission.differences["g16:ldgsts_w1_dram_d4_g16:off:e0"].median == 0.0

    bad_key = "ldgsts_w1_dram_d6_g16"
    bad = {key: list(processes) for key, processes in spacing.items()}
    bad[bad_key] = [
        ProcessReduction(
            process.cell,
            process.repetition,
            process.smid,
            process.issue_by_slot,
            process.completion_by_slot,
            process.gap_on_by_slot,
            ((16.0, 16.0, 16.0, 16.0, 19.0),),
            process.arm_order_pattern,
        )
        for process in bad[bad_key]
    ]
    rejected = admit_gap_matching(bad)
    assert not rejected.admitted
    assert any(f"{bad_key}:off:e4" in failure for failure in rejected.failures)
    malformed_gap = {key: list(processes) for key, processes in spacing.items()}
    malformed_process = malformed_gap[bad_key][0]
    malformed_gap[bad_key][0] = ProcessReduction(
        malformed_process.cell,
        malformed_process.repetition,
        malformed_process.smid,
        malformed_process.issue_by_slot,
        malformed_process.completion_by_slot,
        (("oops", "oops", "oops", "oops", "oops"),),
        malformed_process.gap_off_by_slot,
        malformed_process.arm_order_pattern,
    )
    malformed_admission = admit_gap_matching(malformed_gap)
    assert not malformed_admission.admitted

    assert stable(
        EffectSeries(
            (-0.5, -0.5, 0.0, 0.0, 0.0, 0.5, 0.5),
            0.0,
            0.5,
        )
    )
    assert not stable(EffectSeries((0.0,) * 7, 0.0, 0.500001))
    assert op_eq(summarize_processes([1.0] * 7))
    assert not op_eq(summarize_processes([1.000001] * 7))
    assert not pos(summarize_processes([2.0] * 7))
    assert pos(summarize_processes([2.000001] * 7))
    assert not ordered(summarize_processes([0.0] * 7))
    assert ordered(summarize_processes([0.000001] * 7))

    # Cross-cell effects are formed inside each repetition before the final
    # median/MAD.  The complete W1 fixture must retain the depth-eight route
    # effect and all seven knee families (including the L2 holdout knee).
    w1 = _w1_pskr_fixture()
    pskr = reduce_pskr(w1)
    assert len(pskr.k) == 7
    assert len(pskr.r) == 11
    assert "n8:g0" in pskr.r
    assert pskr.p["l1:g0"].values == (2.0,) * 7
    assert pskr.s["l1:g0"].values == (5.0,) * 7
    assert pskr.k["l1:g0"].values == (3.0,) * 7
    asymmetric_left = _named_series("left", (0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0))
    asymmetric_right = _named_series("right", (0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0))
    asymmetric_difference = difference_series(
        asymmetric_left, asymmetric_right, name="asymmetric"
    )
    assert asymmetric_difference.median == 0.0
    assert asymmetric_left.median - asymmetric_right.median == 10.0
    smid_mismatch = dict(w1)
    smid_mismatch["ldgsts_w1_dram_d4_g0"] = [
        ProcessReduction(
            process.cell,
            process.repetition,
            4 if process.repetition == 3 else process.smid,
            process.issue_by_slot,
            process.completion_by_slot,
            process.gap_on_by_slot,
            process.gap_off_by_slot,
            process.arm_order_pattern,
        )
        for process in w1["ldgsts_w1_dram_d4_g0"]
    ]
    _expect_error(
        lambda: contrast_series(
            smid_mismatch,
            "ldgsts_w1_dram_d4_g0",
            "ldgsts_w1_l1_d4_g0",
        )
    )
    no_pattern = {
        key: [
            ProcessReduction(
                process.cell,
                process.repetition,
                process.smid,
                process.issue_by_slot,
                process.completion_by_slot,
                process.gap_on_by_slot,
                process.gap_off_by_slot,
            )
            for process in processes
        ]
        for key, processes in w1.items()
    }
    _expect_error(
        lambda: contrast_series(
            no_pattern,
            "ldgsts_w1_dram_d4_g0",
            "ldgsts_w1_l1_d4_g0",
        )
    )
    w4_only: dict[str, list[ProcessReduction]] = {}
    for spec in EXPECTED_CELLS:
        if spec.role != "placement":
            continue
        w4_only[spec.key] = [
            _process_reduction_fixture(spec, repetition, 100.0 + repetition)
            for repetition in range(1, DIRECT_REPETITIONS + 1)
        ]
    w1_missing = dict(w1)
    del w1_missing["ldgsts_w1_l1_d4_g0"]
    mixed = {**w1_missing, **w4_only}
    assert "K:l1:g0" in reduce_pskr(mixed).missing
    w4_pskr = reduce_pskr(w4_only, active_warps=4)
    assert knee_key("l1", 0) in w4_pskr.k and not w4_pskr.missing

    # The four mutually exclusive W1 rules and their precedence are evaluated
    # from named synthetic series only; W4/L2 cannot relabel this result.
    clock = _constant_series(3.0, "LDG:K")
    identifications: dict[str, StageResult] = {}
    for label in IDENTIFICATION_LABELS:
        result = classify_identification(
            admission_ok=True,
            clock_knee=clock,
            effects=_identification_effect_fixture(label),
        )
        assert result.status == label
        assert result.identification_label == label
        identifications[label] = result
    ambiguous = classify_identification(
        admission_ok=True,
        clock_knee=clock,
        effects=_identification_effect_fixture("ambiguous"),
    )
    assert ambiguous.status == "valid_ambiguous_no_model"
    assert classify_identification(
        admission_ok=False,
        clock_knee=EffectSeries((3.0,) * 7, 3.0, 0.0),
        effects={},
        admission_failures=("missing route admission",),
    ).status == "invalid_experiment"
    assert classify_identification(
        admission_ok=True,
        clock_knee=_constant_series(0.0, "LDG:K"),
        effects={},
    ).status == "invalid_clock_assay"

    invariant_id = identifications["route_invariant_relief_with_short_gap"]
    invariant_effects = _route_stage_fixture(
        knee={"l1": 3.0, "l2": 3.0, "dram": 3.0},
        a={4: 0.0, 5: 0.0, 6: 0.0},
        b={4: 0.0, 5: 0.0, 6: 0.0},
        total=0.0,
    )
    placement_ok = classify_placement(
        identification=invariant_id,
        admission_ok=True,
        effects_by_slot={slot: invariant_effects for slot in range(4)},
    )
    assert placement_ok.status == "compatible"
    invariant_without_total = {
        family: dict(values) for family, values in invariant_effects.items()
    }
    invariant_without_total["R"].clear()
    assert classify_placement(
        identification=invariant_id,
        admission_ok=True,
        effects_by_slot={slot: invariant_without_total for slot in range(4)},
    ).status == "compatible"
    associated_effects = _route_stage_fixture(
        knee={"l1": 0.0, "l2": 0.0, "dram": 0.0},
        a={4: 0.0, 5: 0.0, 6: 0.0},
        b={4: 0.0, 5: 0.0, 6: 0.0},
        total=0.0,
    )
    placement_associated = classify_placement(
        identification=invariant_id,
        admission_ok=True,
        effects_by_slot={0: invariant_effects, 1: associated_effects, 2: invariant_effects, 3: invariant_effects},
    )
    assert placement_associated.status == "placement_associated"
    assert classify_placement(
        identification=invariant_id,
        admission_ok=True,
        effects_by_slot={slot: associated_effects for slot in range(4)},
    ).status == "falsified"
    assert classify_placement(
        identification=invariant_id,
        admission_ok=False,
        effects_by_slot={},
    ).status == "invalid"
    assert classify_placement(
        identification=invariant_id,
        admission_ok=True,
        effects_by_slot={slot: pskr for slot in range(4)},
    ).status == "invalid"
    assert classify_placement(
        identification=invariant_id,
        admission_ok=True,
        effects_by_slot={slot: invariant_effects for slot in range(4)},
        w1_effects=reduce_pskr(w4_only, active_warps=4),
    ).status == "invalid"
    assert classify_placement(
        identification=ambiguous,
        admission_ok=True,
        effects_by_slot={},
    ).status == "not_applicable"

    sensitive_id = identifications["route_sensitive_issue_response"]
    sensitive_placement_effects = _route_stage_fixture(
        knee={"l1": 3.0, "l2": 3.0, "dram": 3.0},
        a={4: 0.0, 5: 0.0, 6: 1.0},
        b={4: 0.0, 5: 0.0, 6: 1.0},
        total=3.0,
    )
    assert classify_placement(
        identification=sensitive_id,
        admission_ok=True,
        effects_by_slot={slot: sensitive_placement_effects for slot in range(4)},
    ).status == "compatible"
    assert classify_placement(
        identification=identifications["no_material_ldgsts_issue_knee"],
        admission_ok=True,
        effects_by_slot={slot: associated_effects for slot in range(4)},
    ).status == "compatible"
    sensitive_bad_placement = _route_stage_fixture(
        knee={"l1": 3.0, "l2": 3.0, "dram": 3.0},
        a={4: 0.0, 5: 0.0, 6: 0.0},
        b={4: 0.0, 5: 0.0, 6: 0.0},
        total=0.0,
    )
    assert classify_placement(
        identification=sensitive_id,
        admission_ok=True,
        effects_by_slot={slot: sensitive_bad_placement for slot in range(4)},
    ).status == "falsified"

    l2_ok_effects = _l2_stage_fixture(
        knee=3.0,
        a={1: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 0.0},
        b={1: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 0.0},
        total=0.0,
    )
    l2_ok = classify_l2(
        identification=invariant_id,
        admission_ok=True,
        effects=l2_ok_effects,
    )
    assert l2_ok.status == "confirmed"
    l2_falsified = classify_l2(
        identification=invariant_id,
        admission_ok=True,
        effects=_l2_stage_fixture(
            knee=0.0,
            a={1: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 0.0},
            b={1: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 0.0},
            total=0.0,
        ),
    )
    assert l2_falsified.status == "falsified"
    unstable = _l2_stage_fixture(
        knee=3.0,
        a={1: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 0.0},
        b={1: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 0.0},
        total=0.0,
    )
    unstable["A"]["n4:g0"] = _named_series(
        "unstable", (-10.0, -10.0, -10.0, 0.0, 10.0, 10.0, 10.0)
    )
    assert classify_l2(
        identification=invariant_id, admission_ok=True, effects=unstable
    ).status == "inconclusive"
    sensitive_l2 = _l2_stage_fixture(
        knee=3.0,
        a={1: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 8: 1.0},
        b={1: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 8: 1.0},
        total=3.0,
    )
    assert classify_l2(
        identification=sensitive_id,
        admission_ok=True,
        effects=sensitive_l2,
        w1_total_route=_constant_series(3.0, "W1:R6"),
    ).status == "confirmed"
    assert classify_l2(
        identification=sensitive_id,
        admission_ok=True,
        effects=sensitive_l2,
    ).status == "invalid"
    assert classify_l2(
        identification=sensitive_id,
        admission_ok=True,
        effects=sensitive_l2,
        w1_total_route=_constant_series(0.0, "W1:R6"),
    ).status == "falsified"
    assert classify_l2(
        identification=identifications["no_material_ldgsts_issue_knee"],
        admission_ok=True,
        effects=_l2_stage_fixture(
            knee=0.0,
            a={1: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 0.0},
            b={1: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 0.0},
        ),
    ).status == "confirmed"
    assert not stable(EffectSeries((0.0,) * 7, 1.0, 0.0))
    assert classify_l2(
        identification=invariant_id, admission_ok=False, effects={}
    ).status == "invalid"
    assert classify_l2(
        identification=ambiguous, admission_ok=True, effects={}
    ).status == "not_applicable"
    staged = classify_staged(
        identification=invariant_id,
        placement=placement_ok,
        holdout=l2_ok,
    )
    assert staged.successor_eligible
    mismatched_placement = StageResult(
        "compatible", identification_label="route_sensitive_issue_response"
    )
    assert not classify_staged(
        identification=invariant_id,
        placement=mismatched_placement,
        holdout=l2_ok,
    ).successor_eligible
    forged_identification = StageResult(
        invariant_id.status,
        identification_label="route_sensitive_issue_response",
    )
    assert not classify_staged(
        identification=forged_identification,
        placement=placement_ok,
        holdout=l2_ok,
    ).successor_eligible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run focused local checks")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("offline reduction is explicit; pass --self-test")
    self_test()
    print("ldgsts_burst_release analyzer self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
