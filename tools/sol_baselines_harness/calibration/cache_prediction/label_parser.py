"""Parse NCU `--csv` output into a typed `NcuLabelRecord`.

Consumes the CSV emitted by `ncu --csv --section
MemoryWorkloadAnalysis ...` and extracts the read-only sector
hit-rate labels that drive the cache predictor's offline
calibration.

The label record exposes per-launch hit rates as fractions in
`[0.0, 1.0]`. Two metric families per cache level produce the
hit rate:

  * `..._sectors_op_read.sum` (denominator; required) and
    `..._sectors_op_read_hit.sum` (numerator; required) — the
    raw sector counters.
  * `..._sector_op_read_hit_rate.pct` (optional) — the
    percent value NCU reports directly. When present and
    not `n/a`, the parser converts it to a fraction
    (`87.5%` -> `0.875`); when absent or `n/a`, the parser
    reconstructs the rate from `hit_sum / total_sum`.

NCU sometimes reports saturated reads at slightly above
`100%` due to measurement drift. Values in `(1.0, 1.05]`
(fraction units) clamp to `1.0`; values strictly above `1.05`
raise `ncu_label_hit_rate_above_100`.

The combined `..._sector_hit_rate.pct` family (no
`_op_read` qualifier) conflates store traffic under L2's
no-write-allocate write-back policy and biases labels toward
saturation. Records that include any combined-family column
raise `ncu_label_combined_metric_forbidden` BEFORE any
required-column lookup, so a CSV that accidentally mixed both
families also fails closed.

Typed-reason vocabulary:

  * `ncu_label_combined_metric_forbidden` — the CSV header
    contains a combined-family column name; the parser
    refuses to compose a label from such a record.
  * `ncu_label_metric_missing` — a level has no usable
    inputs: both `.pct` and `..._sectors_op_read.sum`
    columns are absent for L1 or L2.
  * `ncu_label_denominator_zero` — the `..._sectors_op_read.sum`
    denominator is zero (n/a label case).
  * `ncu_label_hit_rate_above_100` — the derived hit rate
    exceeds the saturation-clamp ceiling (1.05 in fraction
    units, i.e. 105%).
  * `ncu_label_csv_malformed` — CSV cannot be parsed at all
    (no rows, missing header, type-coerce failure on a
    numeric column).

The parser is pure-function deterministic on its CSV input;
it never invokes NCU. Subprocess execution lives in the
companion `ncu_harvester` module.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


NCU_LABEL_PARSER_TYPED_REASONS: frozenset[str] = frozenset({
    "ncu_label_combined_metric_forbidden",
    "ncu_label_metric_missing",
    "ncu_label_denominator_zero",
    "ncu_label_hit_rate_above_100",
    "ncu_label_csv_malformed",
})


# Required sum-counter columns per level (numerator + denominator
# for the read-only sector hit-rate reconstruction path). The L1
# path filters global loads explicitly via the
# `pipe_lsu_mem_global_op_ld` family so store traffic stays out
# of the numerator and denominator. The L2 path filters by the
# TEX source unit (`srcunit_tex`) so the denominator matches the
# LSU-pipe traffic the NVBit Tier-2 producer can observe —
# instruction-cache cold-launch fetches and constant-cache
# misses route through different source units (icache /
# constant cache) and are NOT counted here. Verified via
# `ncu --query-metrics --chip ga102` on NCU 2026.1.0.0.
NCU_REQUIRED_L1_SECTORS_TOTAL: str = (
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
)
NCU_REQUIRED_L1_SECTORS_HIT: str = (
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"
)
NCU_REQUIRED_L2_SECTORS_TOTAL: str = (
    "lts__t_sectors_srcunit_tex_op_read.sum"
)
NCU_REQUIRED_L2_SECTORS_HIT: str = (
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
)

# Optional `.pct` columns. When present and not `n/a`, the
# parser prefers these over reconstruction; when absent or
# `n/a`, reconstruction from the sum counters is the fallback.
# NCU 2026.1.0.0 does NOT publish a `srcunit_tex_op_read`-
# specific hit-rate.pct column; the L2 hit rate is derived
# from the sum counters via the fallback path. The empty
# string for the L2 .pct column makes the resolver skip the
# fast-path and always reconstruct from `hit / total`.
NCU_OPTIONAL_L1_HIT_RATE_PCT: str = (
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"
)
NCU_OPTIONAL_L2_HIT_RATE_PCT: str = ""

# Combined-family column names (no `_op_read` / `_op_ld`
# qualifier). Records containing any of these are rejected —
# the combined family conflates store traffic and biases the
# label.
NCU_FORBIDDEN_COMBINED_METRICS: tuple[str, ...] = (
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
)

# Saturation-clamp window: NCU sometimes reports `100.x%` on
# saturated workloads due to floating-point measurement drift.
# Fractions in `(1.0, _SATURATION_CLAMP_CEILING]` collapse to
# exactly `1.0`. Strict above-ceiling raises.
_SATURATION_CLAMP_CEILING: float = 1.05


class NcuLabelParserError(Exception):
    """Raised by `parse_ncu_csv` when the input cannot be
    parsed into a clean `NcuLabelRecord`. Carries the
    typed-reason string for downstream typed-unavailable
    bundle composition.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class NcuLabelRecord:
    """Per-launch NCU label record. Hit rates are expressed
    as fractions in `[0.0, 1.0]` — the canonical downstream
    label unit. Provenance fields carry the join-key inputs
    so the resulting record is self-describing.

    L1 sub-cohort is explicitly the global-load family
    (`pipe_lsu_mem_global_op_ld`) so store traffic stays out
    of the numerator and denominator. L2 sub-cohort is the
    sector op_read family (already filters store traffic).
    """

    l1_global_load_hit_rate: float
    l1_global_load_sectors_total: int
    l1_global_load_sectors_hit: int
    l2_read_hit_rate: float
    l2_read_sectors_total: int
    l2_read_sectors_hit: int
    ncu_metric_version: str = "unknown"
    binary_sha256: str = ""
    launch_skip_count: int = -1
    warmup_protocol: str = "unknown"


def _coerce_numeric(value: str, column: str) -> float:
    """Coerce an NCU CSV cell to a finite float. NCU exports
    numeric metrics as strings with possible thousands
    separators or scientific notation; this normalizes the
    common cases. Non-finite values (`nan`, `inf`, `-inf`)
    raise `ncu_label_csv_malformed` — Python's `float()`
    accepts those strings but they would silently corrupt
    label arithmetic downstream.
    """
    if value is None:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} value is None",
        )
    cleaned = value.replace(",", "").strip()
    if cleaned == "" or cleaned.lower() == "n/a":
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} carries non-numeric value "
            f"{value!r}",
        )
    try:
        parsed = float(cleaned)
    except ValueError as exc:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} not float-coercible: "
            f"{value!r} ({exc})",
        ) from exc
    if not math.isfinite(parsed):
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} carries non-finite value "
            f"{value!r}; finite numeric required",
        )
    return parsed


def _coerce_nonneg_integral_counter(
    value: Any,
    column: str,
) -> int:
    """Coerce an NCU sum-counter cell to an exact
    non-negative integer.

    NCU emits sector counters as decimal strings — sometimes
    plain integers (`1024`), sometimes float-formatted
    integers (`1024.0`), sometimes with thousands
    separators (`9,007,199,254,740,993`), and occasionally
    in exact scientific integral form (`1.024e3`). All
    forms must round-trip EXACTLY into the returned
    integer; routing through Python `float()` would
    silently round counters above `2 ** 53` (IEEE-754
    double-precision boundary).

    Validation rules (in order, because comparisons against
    NaN raise `InvalidOperation`):

      1. `None`, empty, and `n/a` raise
         `ncu_label_csv_malformed`.
      2. Whitespace is stripped and thousands separators
         (`,`) are removed before parsing.
      3. The cell is parsed via `decimal.Decimal`. Parse
         failures raise `ncu_label_csv_malformed`.
      4. Non-finite `Decimal` values (`NaN`, `Infinity`,
         `-Infinity`) raise `ncu_label_csv_malformed`
         BEFORE any numeric comparison.
      5. Negative values raise `ncu_label_csv_malformed`.
      6. Non-integral values raise
         `ncu_label_csv_malformed`. Accepts `1024`,
         `1024.0`, and `1.024e3`; rejects `1.5` and
         `1e-1`.
      7. Returns the exact `int` representation.
    """
    if value is None:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} value is None",
        )
    cleaned = str(value).replace(",", "").strip()
    if cleaned == "" or cleaned.lower() == "n/a":
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} carries non-numeric value "
            f"{value!r}",
        )
    try:
        decimal_value = Decimal(cleaned)
    except InvalidOperation as exc:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} not decimal-coercible: "
            f"{value!r} ({exc})",
        ) from exc
    if not decimal_value.is_finite():
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} carries non-finite value "
            f"{value!r}; finite numeric required",
        )
    if decimal_value < 0:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} carries negative value "
            f"{decimal_value!r}; sector counters must be >= 0",
        )
    int_value = int(decimal_value)
    if Decimal(int_value) != decimal_value:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"column {column!r} carries non-integral value "
            f"{decimal_value!r}; sector counters must be "
            f"integral",
        )
    return int_value


def _is_value_present(value: Any) -> bool:
    """A CSV cell counts as "present" when it is not None,
    not empty after stripping, and not the literal `n/a`
    (case-insensitive). Used to decide whether to take the
    `.pct` fast-path or fall back to the sum-counter
    reconstruction.
    """
    if value is None:
        return False
    stripped = str(value).strip()
    if stripped == "":
        return False
    if stripped.lower() == "n/a":
        return False
    return True


def _apply_saturation_clamp(
    fraction: float,
    *,
    column_label: str,
) -> float:
    """Apply the `(1.0, 1.05]` saturation clamp. Fractions in
    that window collapse to `1.0`. Strictly above the ceiling
    raises `ncu_label_hit_rate_above_100`.
    """
    if fraction <= 1.0:
        return fraction
    if fraction <= _SATURATION_CLAMP_CEILING:
        return 1.0
    raise NcuLabelParserError(
        "ncu_label_hit_rate_above_100",
        f"derived hit rate exceeds saturation ceiling "
        f"(value={fraction}, ceiling={_SATURATION_CLAMP_CEILING}, "
        f"source={column_label})",
    )


def _resolve_level_hit_rate(
    row: dict[str, Any],
    fieldnames: list[str],
    *,
    pct_column: str,
    hit_column: str,
    total_column: str,
    level_label: str,
) -> tuple[float, int, int]:
    """Resolve `(hit_rate_fraction, sectors_hit, sectors_total)`
    for one cache level. Tries the `.pct` fast-path first;
    falls back to `hit / total` reconstruction when the `.pct`
    column is absent or carries `n/a`.

    Either path requires the sum-counter columns to be present
    AND the denominator to be non-zero. When both `.pct` AND
    the sum counters are absent for a level, raises
    `ncu_label_metric_missing`.
    """
    # The empty string sentinel for ``pct_column`` means the
    # NCU release does not publish a `.pct` fast-path metric
    # for this level (e.g. the L2 srcunit_tex family on NCU
    # 2026.1.0.0). Skip the fast-path entirely; the
    # sum-counter fallback is the only available source.
    pct_in_header = bool(pct_column) and pct_column in fieldnames
    hit_in_header = hit_column in fieldnames
    total_in_header = total_column in fieldnames

    # Without sum counters there is no fallback path. The
    # `.pct` column alone is insufficient — denominator-zero
    # detection and provenance both depend on the sum
    # counters being present.
    if not hit_in_header or not total_in_header:
        if not pct_in_header:
            raise NcuLabelParserError(
                "ncu_label_metric_missing",
                f"{level_label} level has neither `.pct` nor "
                f"`..._sectors_op_read.sum` /`..._sectors_op_read_hit.sum` "
                f"columns in CSV header; cannot derive hit rate",
            )
        # `.pct` present but sum counters absent — still
        # missing the required denominator/numerator that
        # the label record carries as provenance.
        raise NcuLabelParserError(
            "ncu_label_metric_missing",
            f"{level_label} level has `.pct` but no "
            f"`..._sectors_op_read.sum` /`..._sectors_op_read_hit.sum` "
            f"counters; cannot populate label provenance",
        )

    sectors_total = _coerce_nonneg_integral_counter(
        row[total_column], total_column,
    )
    sectors_hit = _coerce_nonneg_integral_counter(
        row[hit_column], hit_column,
    )
    if sectors_total == 0:
        raise NcuLabelParserError(
            "ncu_label_denominator_zero",
            f"{level_label} level read sector denominator is "
            f"zero (column={total_column}); label is n/a",
        )

    if pct_in_header and _is_value_present(row.get(pct_column)):
        # `.pct` fast path: NCU's reported percent value, in
        # fraction units after the divide-by-100. Negative
        # values are rejected — a percent below zero is not
        # a credible reading.
        pct_value = _coerce_numeric(row[pct_column], pct_column)
        if pct_value < 0:
            raise NcuLabelParserError(
                "ncu_label_csv_malformed",
                f"column {pct_column!r} carries negative "
                f"value {pct_value!r}; hit-rate percent must "
                f"be >= 0",
            )
        fraction = pct_value / 100.0
        return (
            _apply_saturation_clamp(
                fraction, column_label=pct_column,
            ),
            sectors_hit,
            sectors_total,
        )

    # Fallback: reconstruct hit rate from sum counters.
    # Used when NCU omitted the `.pct` column or emitted
    # `n/a` for it.
    fraction = sectors_hit / sectors_total
    return (
        _apply_saturation_clamp(
            fraction,
            column_label=f"{hit_column}/{total_column}",
        ),
        sectors_hit,
        sectors_total,
    )


def parse_ncu_csv(
    csv_text: str,
    *,
    ncu_metric_version: str = "unknown",
    binary_sha256: str = "",
    launch_skip_count: int = -1,
    warmup_protocol: str = "unknown",
) -> NcuLabelRecord:
    """Parse NCU `--csv` output into an `NcuLabelRecord`.

    Per cache level (L1, L2), the parser requires the sum
    counters (`..._sectors_op_read.sum` denominator and
    `..._sectors_op_read_hit.sum` numerator). The `.pct`
    column is OPTIONAL; when present with a numeric value,
    it is preferred over reconstruction. The combined
    `..._sector_hit_rate.pct` family is forbidden.

    Provenance fields (`ncu_metric_version`, `binary_sha256`,
    `launch_skip_count`, `warmup_protocol`) are passed
    through to the returned record; the parser does NOT
    derive these from the CSV itself because they come from
    the harvester's invocation context.

    Typed reasons raised:
      `ncu_label_combined_metric_forbidden`: CSV header
        contains a combined-family column name.
      `ncu_label_metric_missing`: a level has neither
        `.pct` nor sum-counter columns.
      `ncu_label_denominator_zero`: a level's read sector
        denominator is zero.
      `ncu_label_hit_rate_above_100`: a level's derived hit
        rate exceeds 1.05 (the saturation ceiling).
      `ncu_label_csv_malformed`: CSV cannot be parsed at
        all, has no rows, or has a non-numeric value in a
        numeric column.
    """
    if not csv_text or not csv_text.strip():
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            "empty CSV input",
        )
    # Strip NCU's progress / warning lines (`==WARNING==`,
    # `==PROF==`, `==ERROR==`, etc.) AND any target-program
    # stdout that precedes the actual CSV header. NCU writes
    # the profiled binary's stdout (e.g. `D_IN_ADDR=...`) to
    # the same captured stream as the CSV, so the payload
    # extractor walks down to the first CSV header line and
    # raises a typed `no NCU CSV header` error when none is
    # found (the `--launch-skip N` exceeded kernel count
    # surface).
    cleaned_csv = _strip_ncu_marker_lines(csv_text)
    payload_csv = _extract_ncu_csv_payload(cleaned_csv)
    reader = csv.DictReader(io.StringIO(payload_csv))
    if reader.fieldnames is None:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            "CSV header missing",
        )
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

    if _is_long_form_header(fieldnames):
        # NCU's `--metrics`-style output: one metric per row
        # with `Metric Name` / `Metric Value` columns at the
        # end. Pivot to a wide-form row dict before
        # dispatching to the level resolver.
        row = _pivot_long_form_rows_to_wide(rows, fieldnames)
        wide_fieldnames = list(row.keys())
    else:
        # Wide-form: one column per metric, one data row per
        # kernel. Synthetic test fixtures use this shape.
        if not rows:
            raise NcuLabelParserError(
                "ncu_label_csv_malformed",
                "CSV has header but no data rows",
            )
        row = rows[0]
        wide_fieldnames = fieldnames

    # Combined-family rejection: refuse to compose a label
    # from a record that mixes the read-only family with the
    # combined family (or uses combined-family alone).
    combined_present = [
        col for col in NCU_FORBIDDEN_COMBINED_METRICS
        if col in wide_fieldnames
    ]
    if combined_present:
        raise NcuLabelParserError(
            "ncu_label_combined_metric_forbidden",
            f"CSV header contains combined-family column(s) "
            f"{combined_present}; the combined "
            f"`..._sector_hit_rate.pct` family conflates "
            f"store traffic and is forbidden as a calibration "
            f"label source",
        )

    l1_hit_rate, l1_hit, l1_total = _resolve_level_hit_rate(
        row, wide_fieldnames,
        pct_column=NCU_OPTIONAL_L1_HIT_RATE_PCT,
        hit_column=NCU_REQUIRED_L1_SECTORS_HIT,
        total_column=NCU_REQUIRED_L1_SECTORS_TOTAL,
        level_label="L1",
    )
    l2_hit_rate, l2_hit, l2_total = _resolve_level_hit_rate(
        row, wide_fieldnames,
        pct_column=NCU_OPTIONAL_L2_HIT_RATE_PCT,
        hit_column=NCU_REQUIRED_L2_SECTORS_HIT,
        total_column=NCU_REQUIRED_L2_SECTORS_TOTAL,
        level_label="L2",
    )

    return NcuLabelRecord(
        l1_global_load_hit_rate=l1_hit_rate,
        l1_global_load_sectors_total=l1_total,
        l1_global_load_sectors_hit=l1_hit,
        l2_read_hit_rate=l2_hit_rate,
        l2_read_sectors_total=l2_total,
        l2_read_sectors_hit=l2_hit,
        ncu_metric_version=ncu_metric_version,
        binary_sha256=binary_sha256,
        launch_skip_count=launch_skip_count,
        warmup_protocol=warmup_protocol,
    )


def _strip_ncu_marker_lines(csv_text: str) -> str:
    """Drop NCU progress / warning markers that prefix the
    real CSV in tool output. Lines matching `==...==` (e.g.
    `==WARNING==`, `==PROF==`, `==ERROR==`) and trailing
    `Shell cwd ...` lines are removed before CSV parsing.
    """
    kept = []
    for line in csv_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("=="):
            continue
        if stripped.startswith("Shell cwd"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _looks_like_ncu_csv_header(line: str) -> bool:
    """Return True when `line` looks like the NCU CSV header
    row. Two acceptable shapes:

      * long-form: contains both `"ID"` and `"Metric Name"`
        column tokens (NCU 2026.1 `--csv --metrics` output).
      * wide-form: contains at least one of the five pinned
        metric names as a literal substring (synthetic test
        fixtures + future NCU output shapes).

    The check is intentionally permissive: NCU may quote the
    column names differently or reorder them, so a substring
    test suffices.
    """
    if "ID" in line and "Metric Name" in line:
        return True
    for metric in (
        NCU_OPTIONAL_L1_HIT_RATE_PCT,
        NCU_OPTIONAL_L2_HIT_RATE_PCT,
        NCU_REQUIRED_L1_SECTORS_TOTAL,
        NCU_REQUIRED_L1_SECTORS_HIT,
        NCU_REQUIRED_L2_SECTORS_TOTAL,
        NCU_REQUIRED_L2_SECTORS_HIT,
    ):
        # Skip empty-string sentinel constants (e.g.
        # ``NCU_OPTIONAL_L2_HIT_RATE_PCT = ""`` when NCU
        # does not publish a `.pct` column for the chosen
        # L2 metric family). ``"" in line`` is True for any
        # non-empty string, which would otherwise mark
        # every stdout line as a header.
        if metric and metric in line:
            return True
    return False


def _extract_ncu_csv_payload(stripped_text: str) -> str:
    """Slice off any target-program stdout that precedes the
    NCU CSV header. NCU writes the profiled binary's stdout
    AND the CSV table to the same captured stdout stream, so
    lines like `D_IN_ADDR=0x7f...` from the calibration kernels
    appear before the CSV header.

    Walks the input lines top-down, finds the first line that
    matches `_looks_like_ncu_csv_header`, and returns the
    payload from that line onward. Raises
    `ncu_label_csv_malformed` with a `no NCU CSV header`
    message when no header is found — this is the NCU
    `No kernels were profiled` outcome (e.g. `--launch-skip N`
    exceeded the binary's actual kernel launch count).

    `_strip_ncu_marker_lines` MUST run before this — it drops
    `==PROF==` / `==WARNING==` lines that would otherwise
    bracket the binary's stdout.
    """
    lines = stripped_text.splitlines()
    for index, line in enumerate(lines):
        if _looks_like_ncu_csv_header(line):
            return "\n".join(lines[index:])
    raise NcuLabelParserError(
        "ncu_label_csv_malformed",
        "no NCU CSV header in stdout — NCU produced no "
        "kernel profile data. The most common cause is "
        "`--launch-skip N` exceeding the binary's actual "
        "kernel launch count; the target binary must execute "
        "at least `launch_skip_count + 1` kernel launches "
        "for `--launch-count 1` to capture one measured "
        "kernel after warmup.",
    )


def _is_long_form_header(fieldnames: list[str]) -> bool:
    """Detect NCU's long-form CSV by the presence of the
    `Metric Name` / `Metric Value` discriminator columns.
    Wide-form CSVs (one column per metric) carry the metric
    names AS the column headers and never have a literal
    `"Metric Name"` column.
    """
    return "Metric Name" in fieldnames and "Metric Value" in fieldnames


def _pivot_long_form_rows_to_wide(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> dict[str, Any]:
    """Pivot a long-form NCU CSV (one metric per row) into a
    wide-form row dict (`{metric_name: metric_value}`) so the
    downstream level resolver can use the existing wide-form
    code path unchanged.

    Groups rows by the `ID` column (kernel-launch identifier)
    and returns the metrics for `ID == "0"` (the first
    kernel). Subsequent kernels are ignored — the M0 smoke
    contract is one kernel launch per cell.
    """
    if not rows:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            "CSV has header but no data rows",
        )
    metric_name_col = "Metric Name"
    metric_value_col = "Metric Value"
    if (
        metric_name_col not in fieldnames
        or metric_value_col not in fieldnames
    ):
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            f"long-form CSV missing required column(s); "
            f"expected {metric_name_col!r} and "
            f"{metric_value_col!r}; got {fieldnames!r}",
        )
    has_id_col = "ID" in fieldnames
    target_id = "0"
    pivoted: dict[str, Any] = {}
    for row in rows:
        if has_id_col and row.get("ID", "0") != target_id:
            continue
        name = row.get(metric_name_col)
        value = row.get(metric_value_col)
        if name is None:
            continue
        pivoted[name] = value
    if not pivoted:
        raise NcuLabelParserError(
            "ncu_label_csv_malformed",
            "long-form CSV produced no rows for ID=0; "
            "expected at least one metric row",
        )
    return pivoted


__all__ = [
    "NCU_LABEL_PARSER_TYPED_REASONS",
    "NCU_OPTIONAL_L1_HIT_RATE_PCT",
    "NCU_OPTIONAL_L2_HIT_RATE_PCT",
    "NCU_FORBIDDEN_COMBINED_METRICS",
    "NCU_REQUIRED_L1_SECTORS_TOTAL",
    "NCU_REQUIRED_L1_SECTORS_HIT",
    "NCU_REQUIRED_L2_SECTORS_TOTAL",
    "NCU_REQUIRED_L2_SECTORS_HIT",
    "NcuLabelParserError",
    "NcuLabelRecord",
    "parse_ncu_csv",
]
