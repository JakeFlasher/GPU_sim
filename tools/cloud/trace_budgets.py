"""Current per-cell trace budget reader used by the capture tools.

The campaign format is intentionally small: a schema version, an optional
authoring-only ``[defaults]`` table, and explicit ``[[cells]]`` entries. The
capture path never falls back to defaults for an admitted cell.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

VALID_SAMPLING_POLICY_PATTERN = re.compile(
    r"^(?:off|threadblock_stride_[1-9][0-9]*|kernel_limit_[1-9][0-9]*)$"
)

TRACE_VOLUME_EXCEEDED_SIDECAR = "trace_volume_exceeded.json"
NVBIT_SLOWDOWN_BUDGET_EXCEEDED_SIDECAR = "nvbit_slowdown_budget_exceeded.json"
NVBIT_DROP_RATE_EXCEEDED_SIDECAR = "nvbit_drop_rate_exceeded.json"
NVBIT_SLOWDOWN_BUDGET_MAX_RATIO = 1200.0
NVBIT_DROP_RATE_BUDGET_MAX = 0.05
NVBIT_SLOWDOWN_BUDGET_MIN_BASELINE_TIME_S = 0.001


class TraceBudgetErrorKind(Enum):
    TraceBudgetMissing = "TraceBudgetMissing"
    BudgetViolationUntracked = "BudgetViolationUntracked"
    SchemaInvalid = "SchemaInvalid"
    UnknownSamplingPolicy = "UnknownSamplingPolicy"


class TraceBudgetError(Exception):
    def __init__(self, kind: TraceBudgetErrorKind, message: str) -> None:
        self.kind = kind
        self.message = message
        super().__init__(f"{kind.value}: {message}")


@dataclasses.dataclass(frozen=True)
class CellBudget:
    cell_id: str
    max_trace_bytes: int
    max_wall_time_s: int
    sampling_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "max_trace_bytes": self.max_trace_bytes,
            "max_wall_time_s": self.max_wall_time_s,
            "sampling_policy": self.sampling_policy,
        }


@dataclasses.dataclass(frozen=True)
class TraceBudgetsConfig:
    schema_version: int
    defaults: CellBudget | None
    cells: tuple[CellBudget, ...]

    def cells_by_id(self) -> dict[str, CellBudget]:
        return {cell.cell_id: cell for cell in self.cells}


@dataclasses.dataclass(frozen=True)
class BudgetBreachReport:
    cell_id: str
    max_trace_bytes: int
    actual_trace_bytes: int
    max_wall_time_s: int
    actual_wall_time_s: int
    sampling_policy: str
    exceeded_trace_bytes: bool
    exceeded_wall_time: bool

    @property
    def breached(self) -> bool:
        return self.exceeded_trace_bytes or self.exceeded_wall_time

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **dataclasses.asdict(self)}


def _positive_int(value: Any, field: str, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{location} field {field!r} must be a positive integer",
        )
    return value


def _cell(raw: Any, location: str, *, require_id: bool) -> CellBudget:
    if not isinstance(raw, dict):
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{location} must be a table",
        )
    cell_id = raw.get("cell_id", "")
    if require_id and (not isinstance(cell_id, str) or not cell_id.strip()):
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{location} requires a non-empty cell_id",
        )
    if not require_id and "cell_id" in raw:
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{location} must not contain cell_id",
        )
    policy = raw.get("sampling_policy")
    if not isinstance(policy, str) or not VALID_SAMPLING_POLICY_PATTERN.fullmatch(policy):
        raise TraceBudgetError(
            TraceBudgetErrorKind.UnknownSamplingPolicy,
            f"{location} has invalid sampling_policy={policy!r}",
        )
    return CellBudget(
        cell_id=cell_id,
        max_trace_bytes=_positive_int(raw.get("max_trace_bytes"), "max_trace_bytes", location),
        max_wall_time_s=_positive_int(raw.get("max_wall_time_s"), "max_wall_time_s", location),
        sampling_policy=policy,
    )


def load_trace_budgets(path: str | Path) -> TraceBudgetsConfig:
    source = Path(path)
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"cannot read trace budget TOML {source}: {exc}",
        ) from exc
    version = raw.get("schema_version")
    if type(version) is not int or version < 1:
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            "schema_version must be a positive integer",
        )
    defaults = _cell(raw["defaults"], f"{source} [defaults]", require_id=False) if "defaults" in raw else None
    entries = raw.get("cells", [])
    if not isinstance(entries, list):
        raise TraceBudgetError(TraceBudgetErrorKind.SchemaInvalid, "cells must be an array")
    cells = tuple(_cell(entry, f"{source} [[cells]][{i}]", require_id=True) for i, entry in enumerate(entries, 1))
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise TraceBudgetError(TraceBudgetErrorKind.SchemaInvalid, "cell_id values must be unique")
    return TraceBudgetsConfig(version, defaults, cells)


def require_cell_budget(config: TraceBudgetsConfig, cell_id: str) -> CellBudget:
    try:
        return config.cells_by_id()[cell_id]
    except KeyError as exc:
        raise TraceBudgetError(
            TraceBudgetErrorKind.TraceBudgetMissing,
            f"cell {cell_id!r} has no explicit budget entry",
        ) from exc


def check_cell_usage(
    budget: CellBudget, actual_trace_bytes: int, actual_wall_time_s: int
) -> BudgetBreachReport:
    return BudgetBreachReport(
        budget.cell_id,
        budget.max_trace_bytes,
        actual_trace_bytes,
        budget.max_wall_time_s,
        actual_wall_time_s,
        budget.sampling_policy,
        actual_trace_bytes > budget.max_trace_bytes,
        actual_wall_time_s > budget.max_wall_time_s,
    )


def compute_trace_budgets_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_trace_volume_exceeded_sidecar(
    breach: BudgetBreachReport, cell_artifact_dir: str | Path
) -> tuple[Path, str]:
    path = Path(cell_artifact_dir) / TRACE_VOLUME_EXCEEDED_SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(breach.to_dict(), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path, hashlib.sha256(payload.encode()).hexdigest()


def validate_exceeded_cell_has_sidecar(
    breach: BudgetBreachReport, cell_artifact_dir: str | Path
) -> None:
    if not breach.breached:
        return
    path = Path(cell_artifact_dir) / TRACE_VOLUME_EXCEEDED_SIDECAR
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise TraceBudgetError(
            TraceBudgetErrorKind.BudgetViolationUntracked,
            f"cannot read {path}: {exc}",
        ) from exc
    expected = breach.to_dict()
    if payload != expected:
        raise TraceBudgetError(
            TraceBudgetErrorKind.BudgetViolationUntracked,
            f"sidecar {path} does not match the measured breach",
        )

def validate_downstream_trace_budgets_reference(
    downstream_artifact: dict[str, Any], expected_sha256: str
) -> None:
    if downstream_artifact.get("trace_budgets_sha256") != expected_sha256:
        raise TraceBudgetError(
            TraceBudgetErrorKind.TraceBudgetMissing,
            "downstream artifact trace budget reference does not match",
        )
