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


_LONG_COLUMNS = {"Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
_WIDE_IDENTITY_COLUMNS = {
    "ID",
    "Process ID",
    "Process Name",
    "Host Name",
    "Kernel Name",
    "Context",
    "Stream",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
}


def _number(text: str) -> float:
    normalized = text.strip().replace(",", "")
    try:
        value = float(normalized)
    except ValueError as error:
        raise NcuCsvError(f"invalid metric value {text!r}") from error
    if not math.isfinite(value):
        raise NcuCsvError(f"non-finite metric value {text!r}")
    return value


def _kernel_id(text: str) -> int | None:
    raw = text.strip()
    try:
        return int(raw) if raw else None
    except ValueError as error:
        raise NcuCsvError(f"invalid kernel ID {raw!r}") from error


def _parse_long_csv(lines: list[str]) -> list[NcuMetricValue]:
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    values: list[NcuMetricValue] = []
    for row in reader:
        metric_name = (row.get("Metric Name") or "").strip()
        if not metric_name:
            continue
        values.append(
            NcuMetricValue(
                kernel_name=(row.get("Kernel Name") or "").strip(),
                kernel_id=_kernel_id(row.get("ID") or ""),
                metric_name=metric_name,
                metric_unit=(row.get("Metric Unit") or "").strip(),
                metric_value=_number(row.get("Metric Value") or ""),
            )
        )
    return values


def _parse_wide_csv(rows: list[list[str]]) -> list[NcuMetricValue]:
    """Parse NCU 2026 raw-page CSV (header, units, then kernel rows)."""
    header = rows[0]
    if len(rows) < 3:
        raise NcuCsvError("wide NCU CSV requires header, units, and data rows")
    if len(set(header)) != len(header):
        raise NcuCsvError("wide NCU CSV contains duplicate columns")

    units = rows[1]
    if len(units) != len(header):
        raise NcuCsvError("wide NCU CSV units row width does not match header")
    by_name = {name: index for index, name in enumerate(header)}
    kernel_index = by_name["Kernel Name"]
    id_index = by_name.get("ID")
    identity_indices = {
        index for index, name in enumerate(header) if name in _WIDE_IDENTITY_COLUMNS
    }
    if any(units[index].strip() for index in identity_indices):
        raise NcuCsvError("wide NCU CSV is missing its units row")

    values: list[NcuMetricValue] = []
    for row_number, row in enumerate(rows[2:], start=3):
        if len(row) != len(header):
            raise NcuCsvError(
                f"wide NCU CSV row {row_number} width does not match header"
            )
        if not any(field.strip() for field in row):
            continue
        kernel_name = row[kernel_index].strip()
        kernel_id = _kernel_id(row[id_index]) if id_index is not None else None
        for index, metric_name in enumerate(header):
            raw = row[index].strip()
            if index in identity_indices or not metric_name or not raw:
                continue
            try:
                metric_value = _number(raw)
            except NcuCsvError:
                # Wide reports interleave numeric counters with descriptive
                # device attributes and unsupported values such as "n/a".
                continue
            values.append(
                NcuMetricValue(
                    kernel_name=kernel_name,
                    kernel_id=kernel_id,
                    metric_name=metric_name,
                    metric_unit=units[index].strip(),
                    metric_value=metric_value,
                )
            )
    return values


def parse_ncu_csv(text: str) -> list[NcuMetricValue]:
    lines = [line for line in text.splitlines() if line.lstrip().startswith('"')]
    if not lines:
        raise NcuCsvError("no CSV rows found")
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if not rows:
        raise NcuCsvError("no CSV rows found")
    fieldnames = set(rows[0])
    if _LONG_COLUMNS.issubset(fieldnames):
        values = _parse_long_csv(lines)
    elif {"ID", "Kernel Name"}.issubset(fieldnames):
        values = _parse_wide_csv(rows)
    else:
        raise NcuCsvError(
            f"missing NCU CSV columns: {sorted(_LONG_COLUMNS - fieldnames)}"
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
