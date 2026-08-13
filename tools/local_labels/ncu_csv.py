"""Parse Nsight Compute raw-page CSV exports."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class NcuCsvError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NcuMetricValue:
    kernel_name: str
    kernel_id: int | None
    metric_name: str
    metric_unit: str
    metric_value: float


def _number(text: str) -> float:
    normalized = text.strip().replace(",", "")
    try:
        value = float(normalized)
    except ValueError as error:
        raise NcuCsvError(f"invalid metric value {text!r}") from error
    if not math.isfinite(value):
        raise NcuCsvError(f"non-finite metric value {text!r}")
    return value


def parse_ncu_csv(text: str) -> list[NcuMetricValue]:
    lines = [line for line in text.splitlines() if line.lstrip().startswith('"')]
    if not lines:
        raise NcuCsvError("no CSV rows found")
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    fieldnames = set(reader.fieldnames or ())
    required = {"Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
    if not required.issubset(fieldnames):
        raise NcuCsvError(f"missing NCU CSV columns: {sorted(required - fieldnames)}")
    values: list[NcuMetricValue] = []
    for row in reader:
        metric_name = (row.get("Metric Name") or "").strip()
        if not metric_name:
            continue
        raw_id = (row.get("ID") or "").strip()
        try:
            kernel_id = int(raw_id) if raw_id else None
        except ValueError as error:
            raise NcuCsvError(f"invalid kernel ID {raw_id!r}") from error
        values.append(
            NcuMetricValue(
                kernel_name=(row.get("Kernel Name") or "").strip(),
                kernel_id=kernel_id,
                metric_name=metric_name,
                metric_unit=(row.get("Metric Unit") or "").strip(),
                metric_value=_number(row.get("Metric Value") or ""),
            )
        )
    if not values:
        raise NcuCsvError("CSV contains no metric values")
    return values


def load_ncu_csv(path: Path) -> list[NcuMetricValue]:
    return parse_ncu_csv(path.read_text(encoding="utf-8-sig"))


def group_by_metric(values: Iterable[NcuMetricValue]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for value in values:
        grouped.setdefault(value.metric_name, []).append(value.metric_value)
    return grouped
