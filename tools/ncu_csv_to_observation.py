#!/usr/bin/env python3
"""Convert an NCU CSV metric into a post-prediction observation document.

This utility has no prediction input and performs no model selection. It is a
small boundary adapter so NCU values can only enter through the explicit
post-prediction join evaluator.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any


def parse_ncu_csv(path: Path, metric: str, workload: str, source: str = "ncu_csv") -> dict[str, Any]:
    lines = path.read_text().splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.startswith('"ID"'))
    except StopIteration as error:
        raise ValueError("NCU CSV header row is missing") from error
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header_index:]))))
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        try:
            parsed = float(value)
        except ValueError:
            # The first row after the header carries units.
            continue
        values.append(parsed)
    if not values:
        raise ValueError(f"NCU metric {metric!r} has no numeric data rows")
    return {
        "schema": "ncu_observation_v1",
        "metric": metric,
        "source": source,
        "rows": [{"workload": workload, "value": value, "metric": metric, "source": source} for value in values],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--source", default="ncu_csv")
    args = parser.parse_args()
    print(json.dumps(parse_ncu_csv(args.csv, args.metric, args.workload, args.source), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
