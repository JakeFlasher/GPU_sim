#!/usr/bin/env python3
"""Round-0 schema and process reduction for the LDGSTS burst assay.

This module is intentionally local-only. It freezes the 40-cell manifest,
validates one process's paired raw records, retains achieved delays per gap
edge, and exposes the predeclared effect thresholds. Capture-tree discovery,
cross-cell contrasts, staged interpretation, and NCU admission follow only
after the static assay is validated.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


BURSTS = 16_384
WARMUP = 64
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

    require(not isinstance(rows, (str, bytes)), "matrix cells must be a sequence")
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
        require(row.get("arms") == ["enabled", "off"], f"{key}: wrong paired arms")
        require(row.get("bursts") == BURSTS, f"{key}: expected {BURSTS} bursts")
        require(row.get("warmup") == WARMUP, f"{key}: expected warmup {WARMUP}")
        require(
            row.get("direct_repetitions") == DIRECT_REPETITIONS,
            f"{key}: expected seven direct repetitions",
        )
        require(
            row.get("ncu_repeats_min") == NCU_REPEATS_MIN
            and row.get("ncu_repeats_max") == NCU_REPEATS_MAX,
            f"{key}: expected the frozen 3/5 NCU repeat policy",
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


def _gap_bounds(gap: int) -> tuple[float, float]:
    if gap == 0:
        return (0.0, 0.0)
    if gap == 16:
        return (16.0, 38.0)
    if gap == 64:
        return (64.0, 96.0)
    raise AnalysisError(f"unsupported gap sentinel {gap}")


def reduce_process_rows(
    rows: Sequence[Mapping[str, Any]],
    spec: CellSpec,
    repetition: int,
    *,
    expected_bursts: int = BURSTS,
) -> ProcessReduction:
    """Validate and reduce one process without pooling bursts or W4 slots."""

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
                statistics.median(
                    _finite_number(row["issue_on"], "issue_on")
                    - _finite_number(row["issue_off"], "issue_off")
                    for row in slot_rows
                )
            )
        )
        # drain_* is already the interval from issue-end to drain-end.
        completion_by_slot.append(
            float(
                statistics.median(
                    _finite_number(row["drain_on"], "drain_on")
                    - _finite_number(row["drain_off"], "drain_off")
                    for row in slot_rows
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
    )


@dataclass(frozen=True, slots=True)
class EffectSeries:
    """Seven process-level values and their unscaled MAD."""

    values: tuple[float, ...]
    median: float
    mad: float


def summarize_processes(values: Iterable[float]) -> EffectSeries:
    samples = tuple(_finite_number(value, "process effect") for value in values)
    require(len(samples) == DIRECT_REPETITIONS, "effect must contain seven processes")
    center = float(statistics.median(samples))
    mad = float(statistics.median(abs(value - center) for value in samples))
    return EffectSeries(samples, center, mad)


def stable(effect: EffectSeries) -> bool:
    return effect.mad <= 0.5


def op_eq(effect: EffectSeries) -> bool:
    return stable(effect) and abs(effect.median) <= 1.0


def pos(effect: EffectSeries) -> bool:
    return stable(effect) and effect.median > 2.0


def ordered(effect: EffectSeries) -> bool:
    return stable(effect) and effect.median > 0.0


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
        processes = reductions[key]
        if len(processes) != DIRECT_REPETITIONS:
            failures.append(f"{key}: expected seven process reductions")
            continue
        by_repetition: dict[int, ProcessReduction] = {}
        for process in processes:
            if process.cell != key or process.repetition in by_repetition:
                failures.append(f"{key}: invalid or duplicate process reduction")
                continue
            slot_fields = (
                process.issue_by_slot,
                process.completion_by_slot,
                process.gap_on_by_slot,
                process.gap_off_by_slot,
            )
            if any(len(field) != spec.warps for field in slot_fields):
                failures.append(f"{key}: wrong slot cardinality")
                continue
            if any(
                len(edges) != spec.depth - 1
                for field in (process.gap_on_by_slot, process.gap_off_by_slot)
                for edges in field
            ):
                failures.append(f"{key}: wrong retained edge cardinality")
                continue
            low, high = _gap_bounds(spec.gap)
            edge_values = (
                value
                for field in (process.gap_on_by_slot, process.gap_off_by_slot)
                for edges in field
                for value in edges
            )
            if any(not math.isfinite(value) or not low <= value <= high for value in edge_values):
                failures.append(f"{key}: invalid retained achieved gap")
                continue
            by_repetition[process.repetition] = process
        if set(by_repetition) != set(range(1, DIRECT_REPETITIONS + 1)):
            failures.append(f"{key}: repetitions must be exactly 1 through 7")
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
                    "cell": spec.key,
                    "burst": burst,
                    "warp": slot,
                    "arm_order": order,
                    "issue_on": 10 + burst + slot,
                    "issue_off": 2 + burst,
                    "drain_on": 20 + slot,
                    "drain_off": 5,
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


def _expect_error(call: Callable[[], object]) -> None:
    try:
        call()
    except AnalysisError:
        return
    raise AssertionError("expected AnalysisError")


def self_test() -> None:
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

    spec = EXPECTED_BY_KEY["ldgsts_w4_l1_d4_g0"]
    rows = _paired_fixture(spec, 4)
    reduced = reduce_process_rows(rows, spec, 1, expected_bursts=4)
    assert reduced.issue_by_slot == (8.0, 9.0, 10.0, 11.0)
    assert reduced.completion_by_slot == (15.0, 16.0, 17.0, 18.0)
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
        )
        for process in bad[bad_key]
    ]
    rejected = admit_gap_matching(bad)
    assert not rejected.admitted
    assert any(f"{bad_key}:off:e4" in failure for failure in rejected.failures)

    assert stable(EffectSeries((0.0,) * 7, 0.0, 0.5))
    assert not stable(EffectSeries((0.0,) * 7, 0.0, 0.500001))
    assert op_eq(summarize_processes([1.0] * 7))
    assert not op_eq(summarize_processes([1.000001] * 7))
    assert not pos(summarize_processes([2.0] * 7))
    assert pos(summarize_processes([2.000001] * 7))
    assert not ordered(summarize_processes([0.0] * 7))
    assert ordered(summarize_processes([0.000001] * 7))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run focused local checks")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Round 0 is local-only; pass --self-test")
    self_test()
    print("ldgsts_burst_release analyzer self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
