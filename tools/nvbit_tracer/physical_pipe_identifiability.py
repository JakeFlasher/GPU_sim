#!/usr/bin/env python3
"""Determine whether aggregate pipe counts identify dynamic assignments.

This is an identifiability diagnostic, not a hardware route estimator.  It
never chooses one member of an ambiguous feasible set and never consumes NCU
values as labels.  A unique result means only that the supplied candidate sets
and aggregate counts admit one assignment; it is not a silicon route witness.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Any, Iterable, Sequence


MAX_SOLUTIONS = 100_000


class IdentifiabilityRefusal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class DynamicPipeRecord:
    function_uid: int
    pc: int
    warp: int
    ordinal: int
    candidates: tuple[str, ...]

    @classmethod
    def from_object(cls, raw: dict[str, Any]) -> "DynamicPipeRecord":
        if not isinstance(raw, dict):
            raise IdentifiabilityRefusal("record_invalid", "each record must be an object")
        required = ("function_uid", "pc", "warp", "ordinal", "candidates")
        if any(key not in raw for key in required):
            raise IdentifiabilityRefusal("record_invalid", f"record requires {required}")
        raw_candidates = raw["candidates"]
        if not isinstance(raw_candidates, list):
            raise IdentifiabilityRefusal("candidate_domain_invalid", "candidates must be a list")
        if any(not isinstance(candidate, str) or not candidate.strip() for candidate in raw_candidates):
            raise IdentifiabilityRefusal("candidate_domain_invalid", "candidate names must be nonempty strings")
        candidates = tuple(dict.fromkeys(candidate.strip() for candidate in raw_candidates))
        if not candidates:
            raise IdentifiabilityRefusal("candidate_domain_empty", "a dynamic record has no candidates")
        values = [raw[key] for key in required[:4]]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise IdentifiabilityRefusal("record_identity_invalid", "identity fields must be nonnegative integers")
        return cls(*values, candidates)


def _validate_counts(records: Sequence[DynamicPipeRecord], counts: dict[str, int]) -> dict[str, int]:
    if any(not isinstance(pipe, str) or not pipe for pipe in counts):
        raise IdentifiabilityRefusal("aggregate_counts_invalid", "pipe names must be nonempty strings")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise IdentifiabilityRefusal("aggregate_counts_invalid", "aggregate counts must be nonnegative integers")
    candidates = set().union(*(set(record.candidates) for record in records))
    unknown = set(counts) - candidates
    if unknown:
        raise IdentifiabilityRefusal("aggregate_pipe_unknown", f"counts mention unknown pipes: {sorted(unknown)}")
    total = sum(counts.values())
    if total != len(records):
        raise IdentifiabilityRefusal("aggregate_total_mismatch", f"aggregate total {total} != record count {len(records)}")
    return {pipe: counts.get(pipe, 0) for pipe in sorted(candidates)}


def identify(
    records: Sequence[DynamicPipeRecord],
    aggregate_counts: dict[str, int] | None,
    *,
    max_solutions: int = MAX_SOLUTIONS,
) -> dict[str, Any]:
    if not records:
        raise IdentifiabilityRefusal("records_empty", "at least one dynamic record is required")
    if any(not record.candidates for record in records):
        raise IdentifiabilityRefusal("candidate_domain_empty", "a dynamic record has no candidates")
    if max_solutions <= 0 or max_solutions > MAX_SOLUTIONS:
        raise IdentifiabilityRefusal("solution_limit_invalid", f"max_solutions must be in 1..={MAX_SOLUTIONS}")
    if aggregate_counts is None:
        return {
            "schema": "physical_pipe_evidence_v1",
            "status": "route_unobservable",
            "scoring": False,
            "solution_count": None,
            "solutions": [],
            "entropy_bits": None,
            "claims": {"native_route_recovered": False, "aggregate_unique_only": False},
        }
    counts = _validate_counts(records, aggregate_counts)
    ordered = sorted(records, key=lambda record: (len(record.candidates), record.function_uid, record.pc, record.warp, record.ordinal))
    # Keep one extra solution so an exactly-full result is not mislabeled as
    # truncated merely because the search limit was reached.
    solutions: list[tuple[str, ...]] = []
    used: Counter[str] = Counter()
    assignment: list[str] = []

    def search(index: int) -> None:
        if len(solutions) > max_solutions:
            return
        if index == len(ordered):
            if all(used[pipe] == count for pipe, count in counts.items()):
                solutions.append(tuple(assignment))
            return
        record = ordered[index]
        for pipe in record.candidates:
            if used[pipe] >= counts.get(pipe, 0):
                continue
            used[pipe] += 1
            assignment.append(pipe)
            search(index + 1)
            assignment.pop()
            used[pipe] -= 1

    search(0)
    truncated = len(solutions) > max_solutions
    if truncated:
        solutions = solutions[:max_solutions]
    if not solutions:
        status = "route_contract_mismatch"
    elif truncated:
        status = "aggregate_ambiguous"
    elif len(solutions) == 1:
        status = "aggregate_unique"
    else:
        status = "aggregate_ambiguous"
    assignments_by_identity = []
    for solution in solutions:
        assignments_by_identity.append([
            {"function_uid": record.function_uid, "pc": record.pc, "warp": record.warp, "ordinal": record.ordinal, "pipe": pipe}
            for record, pipe in zip(ordered, solution)
        ])
    entropy = None if truncated else log2(len(solutions)) if solutions else None
    return {
        "schema": "physical_pipe_evidence_v1",
        "status": status,
        "scoring": False,
        "record_count": len(records),
        "aggregate_counts": counts,
        "solution_count": None if truncated else len(solutions),
        "solutions_truncated": truncated,
        "solutions": assignments_by_identity,
        "entropy_bits": entropy,
        "claims": {"native_route_recovered": False, "aggregate_unique_only": status == "aggregate_unique"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON object with records and optional aggregate_counts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        records = [DynamicPipeRecord.from_object(record) for record in raw.get("records", [])]
        result = identify(records, raw.get("aggregate_counts"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, IdentifiabilityRefusal, AttributeError) as exc:
        result = {"schema": "physical_pipe_evidence_v1", "status": "refused", "scoring": False, "detail": str(exc)}
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(encoded, end="")
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
