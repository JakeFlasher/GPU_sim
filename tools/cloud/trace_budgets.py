"""Per-cell trace-size and runtime budget validator + enforcement
helper (AC8).

Given a TOML file declaring per-cell limits on trace size, wall-clock
time, and sampling policy, this module exposes:

  * A loader that validates schema at parse time.
  * A strict lookup (`require_cell_budget`) that returns ONLY an
    explicit per-cell `[[cells]]` entry. Undeclared cells raise
    `TraceBudgetMissing` even when `[defaults]` is present — the
    defaults block exists solely as a TOML-authoring template; it
    never papers over a missing enforcement entry.
  * A pure breach-detection check and a JSON sidecar writer that
    emits `trace_volume_exceeded.json` (plan-mandated filename)
    inside the cell's artifact directory.
  * A companion validator that parses the sidecar and asserts that
    it names the cell and the sampling policy that kicked in, so
    downstream tooling cannot satisfy the gate with a blank or
    mismatched file.
  * A downstream-reference validator that refuses to honour a
    collection artifact whose `trace_budgets_sha256` is missing or
    does not match the canonical SHA-256 of the committed budgets
    TOML.

`sampling_policy` is a regex-bounded enum:

    sampling_policy := "off"
                     | "threadblock_stride_" <positive_int>
                     | "kernel_limit_" <positive_int>

Any value outside that grammar raises `UnknownSamplingPolicy` at
load time. Exceeded cells MUST produce a sidecar named exactly
`trace_volume_exceeded.json` inside the cell's artifact directory;
collection artifacts that omit it (or whose sidecar's content
contradicts the declared budget or the breach it is supposed to
record) fail the companion validator with
`BudgetViolationUntracked`.

Public surface:

    from trace_budgets import (
        CellBudget,
        TraceBudgetsConfig,
        BudgetBreachReport,
        TraceBudgetError,
        TraceBudgetErrorKind,
        VALID_SAMPLING_POLICY_PATTERN,
        load_trace_budgets,
        require_cell_budget,
        check_cell_usage,
        write_trace_volume_exceeded_sidecar,
        validate_exceeded_cell_has_sidecar,
        validate_downstream_trace_budgets_reference,
        compute_trace_budgets_sha256,
    )
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any


# Regex bound on the `sampling_policy` string. Matches the three
# grammars declared by AC8: the literal `"off"`, the stride form
# `threadblock_stride_<N>` with N ≥ 1, and the kernel-count cap
# `kernel_limit_<N>` with N ≥ 1. Negative ints, zero, and trailing
# garbage are rejected.
VALID_SAMPLING_POLICY_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:off|threadblock_stride_[1-9][0-9]*|kernel_limit_[1-9][0-9]*)$"
)


class TraceBudgetErrorKind(enum.Enum):
    TraceBudgetMissing = "TraceBudgetMissing"
    BudgetViolationUntracked = "BudgetViolationUntracked"
    SchemaInvalid = "SchemaInvalid"
    UnknownSamplingPolicy = "UnknownSamplingPolicy"


class TraceBudgetError(Exception):
    def __init__(self, kind: TraceBudgetErrorKind, message: str) -> None:
        super().__init__(f"{kind.value}: {message}")
        self.kind = kind
        self.message = message


@dataclasses.dataclass(frozen=True)
class CellBudget:
    """Per-cell budget declaration.

    `cell_id` is empty only on the `[defaults]` authoring-template
    entry, which exists to help TOML authors copy a baseline when
    adding a new `[[cells]]` row. `[defaults]` is NEVER used as a
    fallback for enforcement — `require_cell_budget` refuses any
    cell without an explicit `[[cells]]` entry per AC8.1.
    """

    cell_id: str
    max_trace_bytes: int
    max_wall_time_s: int
    sampling_policy: str

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "max_trace_bytes": self.max_trace_bytes,
            "max_wall_time_s": self.max_wall_time_s,
            "sampling_policy": self.sampling_policy,
        }


@dataclasses.dataclass(frozen=True)
class TraceBudgetsConfig:
    """Parsed `trace_budgets.toml`."""

    schema_version: int
    defaults: CellBudget | None
    cells: tuple[CellBudget, ...]

    def cells_by_id(self) -> dict[str, CellBudget]:
        return {c.cell_id: c for c in self.cells}


@dataclasses.dataclass(frozen=True)
class BudgetBreachReport:
    """Pure summary of actual-vs-budget. `breached` is True iff
    either of the two exceeded flags is True. Written verbatim into
    the `trace_volume_exceeded.json` sidecar.
    """

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

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "cell_id": self.cell_id,
            "max_trace_bytes": self.max_trace_bytes,
            "actual_trace_bytes": self.actual_trace_bytes,
            "max_wall_time_s": self.max_wall_time_s,
            "actual_wall_time_s": self.actual_wall_time_s,
            "sampling_policy": self.sampling_policy,
            "exceeded_trace_bytes": self.exceeded_trace_bytes,
            "exceeded_wall_time": self.exceeded_wall_time,
        }


def compute_trace_budgets_sha256(path: str | Path) -> str:
    """SHA-256 of the raw `trace_budgets.toml` bytes. Stable across
    repeated loads because the file bytes are hashed directly (no
    parsing)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_sampling_policy(
    raw: Any, location: str
) -> str:
    if not isinstance(raw, str):
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{location} field 'sampling_policy' must be a string, "
            f"got {type(raw).__name__}",
        )
    if not VALID_SAMPLING_POLICY_PATTERN.match(raw):
        raise TraceBudgetError(
            TraceBudgetErrorKind.UnknownSamplingPolicy,
            f"{location} field 'sampling_policy'={raw!r} is not a "
            f"valid policy. Allowed: 'off', 'threadblock_stride_<N>', "
            f"or 'kernel_limit_<N>' with N >= 1.",
        )
    return raw


def _require_positive_int(
    container: dict[str, Any], key: str, location: str
) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{location} field {key!r} must be an int, got "
            f"{type(value).__name__}",
        )
    if value <= 0:
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{location} field {key!r} must be > 0, got {value}",
        )
    return value


def _parse_cell_entry(
    row: Any, location: str, *, require_cell_id: bool
) -> CellBudget:
    if not isinstance(row, dict):
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{location} must be a table, got {type(row).__name__}",
        )
    cell_id = row.get("cell_id", "")
    if require_cell_id:
        if not isinstance(cell_id, str) or not cell_id.strip():
            raise TraceBudgetError(
                TraceBudgetErrorKind.SchemaInvalid,
                f"{location} missing required non-empty 'cell_id' string",
            )
    else:
        # `[defaults]` entries do not carry cell_id.
        if "cell_id" in row:
            raise TraceBudgetError(
                TraceBudgetErrorKind.SchemaInvalid,
                f"{location} must not carry 'cell_id' — defaults apply "
                f"to cells that omit their own budget entry",
            )
        cell_id = ""

    max_trace_bytes = _require_positive_int(
        row, "max_trace_bytes", location
    )
    max_wall_time_s = _require_positive_int(
        row, "max_wall_time_s", location
    )
    sampling_policy = _validate_sampling_policy(
        row.get("sampling_policy"),
        location,
    )
    return CellBudget(
        cell_id=cell_id,
        max_trace_bytes=max_trace_bytes,
        max_wall_time_s=max_wall_time_s,
        sampling_policy=sampling_policy,
    )


def load_trace_budgets(path: str | Path) -> TraceBudgetsConfig:
    """Parse and validate a `trace_budgets.toml` file.

    Enforces at load time:
      * `schema_version = 1` at the top of the file.
      * An optional `[defaults]` table with
        `max_trace_bytes`, `max_wall_time_s`, `sampling_policy`.
      * An optional `[[cells]]` array of entries with the same
        three fields plus a non-empty `cell_id`.
      * `max_trace_bytes` and `max_wall_time_s` positive ints.
      * `sampling_policy` in `{off, threadblock_stride_<N>,
        kernel_limit_<N>}` (N >= 1).
      * Cell IDs unique across the file.

    Raises `TraceBudgetError(SchemaInvalid)` on shape/type
    violations and `TraceBudgetError(UnknownSamplingPolicy)` on
    an off-enum `sampling_policy` value.
    """
    p = Path(path)
    try:
        with p.open("rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError as exc:
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"trace-budgets TOML not found: {p}",
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"trace-budgets TOML parse error in {p}: {exc}",
        ) from exc

    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(
        schema_version, bool
    ) or schema_version < 1:
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{p} missing or invalid 'schema_version' (must be int >= 1)",
        )

    defaults: CellBudget | None = None
    if "defaults" in raw:
        defaults_raw = raw["defaults"]
        if not isinstance(defaults_raw, dict):
            raise TraceBudgetError(
                TraceBudgetErrorKind.SchemaInvalid,
                f"{p} [defaults] must be a table, got "
                f"{type(defaults_raw).__name__}",
            )
        defaults = _parse_cell_entry(
            defaults_raw, f"{p} [defaults]", require_cell_id=False
        )

    cells_raw = raw.get("cells", [])
    if not isinstance(cells_raw, list):
        raise TraceBudgetError(
            TraceBudgetErrorKind.SchemaInvalid,
            f"{p} 'cells' must be an array of tables",
        )

    cells: list[CellBudget] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(cells_raw, start=1):
        location = f"{p} [[cells]][{idx}]"
        parsed = _parse_cell_entry(entry, location, require_cell_id=True)
        if parsed.cell_id in seen_ids:
            raise TraceBudgetError(
                TraceBudgetErrorKind.SchemaInvalid,
                f"{location} duplicate cell_id {parsed.cell_id!r}",
            )
        seen_ids.add(parsed.cell_id)
        cells.append(parsed)

    return TraceBudgetsConfig(
        schema_version=schema_version,
        defaults=defaults,
        cells=tuple(cells),
    )


def require_cell_budget(
    config: TraceBudgetsConfig, cell_id: str
) -> CellBudget:
    """Return the explicit per-cell budget for `cell_id`.

    AC8.1 requires every admitted cell to have an upfront
    per-cell declaration; undeclared cells MUST raise
    `TraceBudgetMissing`. The `[defaults]` block exists as a
    TOML-authoring template and is deliberately NOT consulted
    here — otherwise a new SolBench cell could slip through the
    admission gate without an explicit upfront budget.

    Callers that want the defaults block (e.g. to template a new
    cell entry) should read `config.defaults` directly.
    """
    cells = config.cells_by_id()
    if cell_id in cells:
        return cells[cell_id]
    raise TraceBudgetError(
        TraceBudgetErrorKind.TraceBudgetMissing,
        f"cell {cell_id!r} has no [[cells]] entry in the "
        f"trace-budgets config. AC8.1 requires every admitted "
        f"cell to declare its budget upfront; the [defaults] "
        f"block is an authoring template only and is never used "
        f"for enforcement.",
    )


def check_cell_usage(
    budget: CellBudget,
    actual_trace_bytes: int,
    actual_wall_time_s: int,
) -> BudgetBreachReport:
    """Compare actual usage against the cell's declared budget.

    Pure: returns a `BudgetBreachReport` whose `breached` property
    is True iff either limit was exceeded. Callers decide whether
    to write a sidecar (typically only when `breached` is True).
    """
    exceeded_bytes = actual_trace_bytes > budget.max_trace_bytes
    exceeded_time = actual_wall_time_s > budget.max_wall_time_s
    return BudgetBreachReport(
        cell_id=budget.cell_id,
        max_trace_bytes=budget.max_trace_bytes,
        actual_trace_bytes=actual_trace_bytes,
        max_wall_time_s=budget.max_wall_time_s,
        actual_wall_time_s=actual_wall_time_s,
        sampling_policy=budget.sampling_policy,
        exceeded_trace_bytes=exceeded_bytes,
        exceeded_wall_time=exceeded_time,
    )


# Plan-mandated sidecar filename. AC8.2 names the file
# `trace_volume_exceeded.json`; callers scope uniqueness by passing
# the cell's own artifact directory, so there is no need to mangle
# the filename with a cell_id prefix.
TRACE_VOLUME_EXCEEDED_SIDECAR: str = "trace_volume_exceeded.json"

# C-AC-H sidecar filenames (Loop 2 R10). Two new typed reasons in
# the trace-budget taxonomy family, both pinned in the
# original-plan exclusion-taxonomy list:
#   - `nvbit_slowdown_budget_exceeded`: emitted when an NVBit-traced
#     run's wall-clock time vs. its non-traced baseline crosses the
#     declared slowdown budget. The matching schema records the
#     declared budget, observed slowdown, and reference baseline.
#   - `nvbit_drop_rate_exceeded`: emitted when the producer-side drop
#     counters indicate the trace is missing more than the declared
#     fraction of expected records. The matching schema's description
#     enumerates the four drop-counter sources (NVBit channel
#     overflow + ring-buffer loss + decoder gaps + expected-record
#     mismatch).
NVBIT_SLOWDOWN_BUDGET_EXCEEDED_SIDECAR: str = "nvbit_slowdown_budget_exceeded.json"
NVBIT_DROP_RATE_EXCEEDED_SIDECAR: str = "nvbit_drop_rate_exceeded.json"

# C-AC-H fixed budget thresholds (Loop 2 R11). These pin the
# Admissibility budgets for NVBit traces:
#   - slowdown ratio: actual_wall_time_s / baseline_wall_time_s
#     must stay at or below the global cap. A ratio strictly
#     greater than the cap trips the slowdown sidecar.
#   - drop rate: 1 - admitted_record_count / expected_record_count
#     must stay at or below 0.05 (5%). A ratio strictly greater
#     than 0.05 trips the drop-rate sidecar.
# These are pinned constants rather than per-row config fields:
# the cap value is fixed and applies to every row; row-level
# overrides would re-introduce the dormant-writer bug Codex
# flagged in the R10 review. Plan evolutions widen the GLOBAL
# constant, never add a per-row override.
#
# Cap history:
#   - Original plan: 200.0 (the original-plan threshold; pinned
#     for the bulk of the R48 work).
#   - Round-25 (historical) user-sanctioned widening: 200.0 → 400.0.
#     Authorized by the in-session AskUserQuestion at round-25 entry
#     ("Sanction wider NVBit slowdown budget for l2_prefetch_hint_sweep").
#     Recorded as a Plan Evolution Log entry in the round-25
#     goal-tracker mutable section. Two slowdown observations
#     drove this evolution and MUST stay distinct in this prose:
#       (a) Round-9 BLOCKED attempt under the original-plan 200x
#           cap measured 335.6x and triggered the budget-exceeded
#           sidecar; that is the historical observation the
#           original cap rejected.
#       (b) Round-25 LIVE ADMITTED recapture under the (historical)
#           400x cap measured 381.4595267441686x (≈ 381.5x) at
#           capture time. The historical 400x cap admitted that
#           observation under the live-evidence harness with
#           ≈ 4.9% headroom over that historical value
#           ((400 - 381.4595) / 381.4595 ≈ 0.0486). The trace JSON
#           has since been RE-CAPTURED (see the current-live entry
#           below); the 381.4595x bytes are superseded on disk.
#   - Round-4 (sm120_l2_observability_30_cell_cohort loop)
#     user-sanctioned widening: 400.0 → 1200.0. Authorized by the
#     in-session AskUserQuestion when l2_prefetch_hint_sweep /
#     trace_l2_accuracy_xl exceeded the (historical) 400x cap on
#     sm_120 (RTX 5060 laptop) at `actual_slowdown_ratio = 1010.85x`,
#     producing `nvbit_slowdown_budget_exceeded.json` and no usable
#     trace. The widened 1200x cap admits the sm_120 trace with ≈
#     18.7% headroom over the observed 1010.85x value. Recorded as
#     a Plan Evolution Log entry in the round-4 goal-tracker mutable
#     section; the active cap is now 1200x. The round-25 references
#     above are HISTORICAL.
#   - Current sm_86 live trace (canonical-geometry relabel era): the
#     committed trace JSON at
#     `workloads/simbench_bw/results/l2_prefetch_hint_sweep/rtx_3090_ti/sm_86/trace_trace_l2_accuracy_xl.json`
#     now records `actual_slowdown_ratio = 434.72113981734213x`
#     (≈ 434.7x), the sm_86 re-capture that superseded the
#     historical 381.4595x bytes of observation (b). The active
#     1200x cap admits it with ≈ 176.0% headroom over the live
#     value ((1200 - 434.7211) / 434.7211 ≈ 1.7604).
NVBIT_SLOWDOWN_BUDGET_MAX_RATIO: float = 1200.0
NVBIT_DROP_RATE_BUDGET_MAX: float = 0.05

# Minimum un-instrumented baseline wall-clock time, in seconds, below
# which the slowdown-ratio admissibility check is suppressed. The
# `NVBIT_SLOWDOWN_BUDGET_MAX_RATIO` cap (currently 1200.0 post-R4; see
# the cap-history paragraph above for the original-plan 200.0 → 400.0
# → 1200.0 widening provenance) is a relative measure
# (`actual_wall_time_s / baseline_wall_time_s`); when the baseline is
# microsecond-fast (e.g. 0.000127s for `dram_stream_stride_mix/trace_00`
# on Blackwell GB206; 0.000138s for the same shape on Ampere GA102),
# normal per-instruction NVBit instrumentation overhead absolutely
# dominates and the ratio inflates well past the runtime cap
# (~14000x on sm_86, ~19000x on sm_120 — far above either the
# original-plan 200x, the historical post-R25 400x, or the active
# post-R4 1200x cap) without indicating a real producer-side
# regression. The threshold establishes a meaningful-baseline floor:
# only kernels whose un-instrumented baseline meets or exceeds 1.0 ms
# (1000 microseconds) are subject to the
# `NVBIT_SLOWDOWN_BUDGET_MAX_RATIO` ratio cap. Below the floor, the
# slowdown sidecar MUST NOT be emitted by the producer and MUST NOT
# be required by the verifier — both surfaces honor the same constant
# for symmetry. Slow-baseline kernels with long un-instrumented wall
# times remain fully gated by the runtime
# `NVBIT_SLOWDOWN_BUDGET_MAX_RATIO` cap (1200x post-R4).
NVBIT_SLOWDOWN_BUDGET_MIN_BASELINE_TIME_S: float = 0.001


def write_trace_volume_exceeded_sidecar(
    breach: BudgetBreachReport, cell_artifact_dir: str | Path
) -> tuple[Path, str]:
    """Write the `trace_volume_exceeded.json` sidecar inside the
    cell's artifact directory.

    Returns `(path, sha256)` of the written file. Callers typically
    only invoke this when `breach.breached` is True, but the
    function itself does not enforce that — tests emit sidecars
    unconditionally to exercise the writer.

    The emitted JSON carries the full breach record, including
    `sampling_policy` (the plan-mandated "name the policy that
    kicked in" field) and both exceeded flags.
    """
    directory = Path(cell_artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRACE_VOLUME_EXCEEDED_SIDECAR
    serialized = json.dumps(breach.to_dict(), indent=2) + "\n"
    path.write_text(serialized)
    return path, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _violation_untracked(detail: str, cell_id: str) -> TraceBudgetError:
    return TraceBudgetError(
        TraceBudgetErrorKind.BudgetViolationUntracked,
        f"cell {cell_id!r}: {detail}",
    )


def validate_exceeded_cell_has_sidecar(
    breach: BudgetBreachReport, cell_artifact_dir: str | Path
) -> None:
    """For a breached cell, assert that the plan-mandated
    `trace_volume_exceeded.json` sidecar exists in
    `cell_artifact_dir` AND its contents match the breach record.

    AC8.2 requires the sidecar to name the policy that kicked in;
    this function enforces it by parsing the JSON and asserting
    that `schema_version == 1`, `cell_id` matches, the declared
    `sampling_policy` matches, all four `max_*` / `actual_*`
    budget fields match, and at least one `exceeded_*` flag is
    True (a sidecar for a "breached" cell that records neither
    breach is a self-contradiction).

    A non-breached cell is a no-op regardless of sidecar presence.

    Any missing / malformed / mismatched sidecar raises
    `TraceBudgetError(BudgetViolationUntracked)` with a message
    naming the specific discrepancy.
    """
    if not breach.breached:
        return
    path = Path(cell_artifact_dir) / TRACE_VOLUME_EXCEEDED_SIDECAR
    if not path.is_file():
        raise _violation_untracked(
            f"trace-budget breach (max_trace_bytes={breach.max_trace_bytes}, "
            f"actual={breach.actual_trace_bytes}; "
            f"max_wall_time_s={breach.max_wall_time_s}, "
            f"actual={breach.actual_wall_time_s}) has no sidecar "
            f"at {path}",
            breach.cell_id,
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise _violation_untracked(
            f"sidecar at {path} is not valid JSON: {exc}",
            breach.cell_id,
        ) from exc
    if not isinstance(payload, dict):
        raise _violation_untracked(
            f"sidecar at {path} root is not a JSON object",
            breach.cell_id,
        )
    if payload.get("schema_version") != 1:
        raise _violation_untracked(
            f"sidecar at {path} has schema_version="
            f"{payload.get('schema_version')!r}, expected 1",
            breach.cell_id,
        )
    if payload.get("cell_id") != breach.cell_id:
        raise _violation_untracked(
            f"sidecar at {path} has cell_id="
            f"{payload.get('cell_id')!r}, expected "
            f"{breach.cell_id!r}",
            breach.cell_id,
        )
    if payload.get("sampling_policy") != breach.sampling_policy:
        raise _violation_untracked(
            f"sidecar at {path} has sampling_policy="
            f"{payload.get('sampling_policy')!r}, expected "
            f"{breach.sampling_policy!r} (the policy declared in "
            f"trace_budgets.toml for this cell — AC8.2 requires "
            f"the sidecar to name the policy that kicked in)",
            breach.cell_id,
        )
    for field, expected in (
        ("max_trace_bytes", breach.max_trace_bytes),
        ("actual_trace_bytes", breach.actual_trace_bytes),
        ("max_wall_time_s", breach.max_wall_time_s),
        ("actual_wall_time_s", breach.actual_wall_time_s),
    ):
        if payload.get(field) != expected:
            raise _violation_untracked(
                f"sidecar at {path} has {field}="
                f"{payload.get(field)!r}, expected {expected!r}",
                breach.cell_id,
            )
    recorded_bytes_flag = payload.get("exceeded_trace_bytes")
    recorded_time_flag = payload.get("exceeded_wall_time")
    if not isinstance(recorded_bytes_flag, bool) or not isinstance(
        recorded_time_flag, bool
    ):
        raise _violation_untracked(
            f"sidecar at {path} is missing bool exceeded_* flags "
            f"(got exceeded_trace_bytes={recorded_bytes_flag!r}, "
            f"exceeded_wall_time={recorded_time_flag!r})",
            breach.cell_id,
        )
    if not (recorded_bytes_flag or recorded_time_flag):
        raise _violation_untracked(
            f"sidecar at {path} records neither flag True, but the "
            f"runtime breach says this cell exceeded its budget — "
            f"the sidecar cannot disclaim the breach it is meant "
            f"to track",
            breach.cell_id,
        )
    if recorded_bytes_flag != breach.exceeded_trace_bytes:
        raise _violation_untracked(
            f"sidecar at {path} has exceeded_trace_bytes="
            f"{recorded_bytes_flag!r}, expected "
            f"{breach.exceeded_trace_bytes!r}",
            breach.cell_id,
        )
    if recorded_time_flag != breach.exceeded_wall_time:
        raise _violation_untracked(
            f"sidecar at {path} has exceeded_wall_time="
            f"{recorded_time_flag!r}, expected "
            f"{breach.exceeded_wall_time!r}",
            breach.cell_id,
        )


def validate_downstream_trace_budgets_reference(
    downstream_artifact: dict[str, Any],
    expected_sha256: str,
) -> None:
    """Verify `downstream_artifact["trace_budgets_sha256"]` matches
    `expected_sha256`.

    Raises `TraceBudgetError(TraceBudgetMissing)` if the reference
    is missing, not a non-empty string, or mismatched. Returns None
    on success.
    """
    key = "trace_budgets_sha256"
    actual = downstream_artifact.get(key)
    if not isinstance(actual, str) or not actual.strip():
        raise TraceBudgetError(
            TraceBudgetErrorKind.TraceBudgetMissing,
            f"downstream artifact has no {key!r} reference",
        )
    if actual != expected_sha256:
        raise TraceBudgetError(
            TraceBudgetErrorKind.TraceBudgetMissing,
            f"{key} mismatch (artifact={actual!r}, "
            f"expected={expected_sha256!r})",
        )


__all__ = [
    "BudgetBreachReport",
    "CellBudget",
    "NVBIT_DROP_RATE_BUDGET_MAX",
    "NVBIT_DROP_RATE_EXCEEDED_SIDECAR",
    "NVBIT_SLOWDOWN_BUDGET_EXCEEDED_SIDECAR",
    "NVBIT_SLOWDOWN_BUDGET_MAX_RATIO",
    "NVBIT_SLOWDOWN_BUDGET_MIN_BASELINE_TIME_S",
    "TRACE_VOLUME_EXCEEDED_SIDECAR",
    "TraceBudgetError",
    "TraceBudgetErrorKind",
    "TraceBudgetsConfig",
    "VALID_SAMPLING_POLICY_PATTERN",
    "check_cell_usage",
    "compute_trace_budgets_sha256",
    "load_trace_budgets",
    "require_cell_budget",
    "validate_downstream_trace_budgets_reference",
    "validate_exceeded_cell_has_sidecar",
    "write_trace_volume_exceeded_sidecar",
]
