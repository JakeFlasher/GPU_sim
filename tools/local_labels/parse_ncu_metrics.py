"""Safe parser for Nsight Compute details-page CSV output.

The label driver imports a report with ``ncu --import <report> --page
details --csv ...`` and pipes the output to this parser. The details
page emits one row per (kernel-launch, section, metric) tuple with a
documented column layout where the metric value lives in the column
whose header is ``Metric Value`` (the unit lives in ``Metric Unit``).
The parser:

- reads the CSV header to locate the metric-name and metric-value
  columns by name (not by ordinal), so changes in the leading
  identifier columns (process / context / stream / kernel-name / device)
  do not silently shift the value column;
- selects rows by exact metric name from the metric-name column;
- builds a launch-identity key for every selected row from the ``ID``
  column when present, falling back to the tuple of leading
  identifier columns (``Process ID``, ``Context``, ``Stream``,
  ``Device``, ``Kernel Name``, ``Kernel Time``) when ``ID`` is absent;
- admits same-value duplicates only within a single launch identity
  (which is what NCU 2026.1+ produces when the same metric appears in
  multiple sections of one launch);
- rejects ANY appearance of more than one launch identity for any
  requested metric — even when the values match — because that is the
  exact launch-filter regression signal (multiple kernel launches
  imported without ``--launch-skip`` / ``--launch-count 1``);
- rejects missing metrics, non-numeric values, and percentage values
  outside ``[0, 100]``;
- returns each metric as a fraction in ``[0, 1]``.

R10 introduced a same-value tolerance for the multi-section
repetition NCU 2026.1+ emits (e.g., ``lts__t_sector_hit_rate.pct``
appears in both "Command line profiler metrics" and "Memory Workload
Analysis"). R11 tightens the guard so the tolerance applies ONLY
within a single launch identity. Two distinct launches that happen to
return the same numeric value are still rejected — R10's plain
value-match check would have admitted that case and silently masked a
launch-filter regression.

R10 also switched the import page from ``raw`` to ``details``
because NCU 2026.1.1.0 restricts ``--print-metric-name name`` to the
details page (``--page raw --csv`` without that flag emits wide-format
CSV with one column per metric instead of one row per metric).

A later amendment (raw-counter ratio fallback) handles a sm_120 / Blackwell
consumer defect class where ``lts__t_sector_hit_rate.pct`` intermittently
returns values far above the existing saturation-rounding window
``[100, 105]`` (observed 115.6%, 202.01%, 217.66% on
``silu_fwd_bf16_sm80/correctness_shape``). The values are not ordinary
quantization rounding; they are a different defect class (likely
ratio aggregation / replay normalization / collection-path overcounting).
Clamping them to ``1.0`` would silently mask a measurement defect and
likely inflate variance. Instead, when the caller requests a ``.pct``
metric registered in ``_PCT_TO_RAW_COUNTER_PAIR``, the parser also
collects the matching ``.sum`` lookup-hit / lookup-miss counter pair
from the same CSV; on out-of-range ``.pct`` it derives
``hit / (hit + miss)`` (mathematically bounded in ``[0, 1]``) and
returns the derived value with a new typed diagnostic
``ncu_lts_pct_overcount_raw_counter_ratio_used`` that records every
fallback event. The public metric identity returned to the caller is
preserved (the contract pin on ``lts__t_sector_hit_rate.pct`` in
``workloads/simbench_bw/comparison_contract.toml`` is intact); only
the derivation path changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor as ncu_executor


DEFAULT_METRIC_NAME_COLUMN = "Metric Name"
DEFAULT_METRIC_VALUE_COLUMN = "Metric Value"
_OBSERVATION_COLUMNS = (
    "ID", "Process ID", "Process Name", "Host Name", "Context", "Stream",
    "Device", "Kernel Name", "Block Size", "Grid Size",
)
_REQUIRED_OBSERVATION_COLUMNS = (
    "ID", "Kernel Name", "Block Size", "Grid Size", "Device",
)
_LAUNCH_SET_IDENTITY_COLUMNS = (
    "ID", "Process ID", "Context", "Stream", "Device", "Kernel Name",
    "Block Size", "Grid Size",
)


@dataclass
class _LaunchCsvGroup:
    identity: dict[str, str]
    rows: list[list[str]]


def _split_launch_set_csv(
    text: str, *, expected_launch_count: int,
    requested_launch_skip: int, requested_launch_count: int,
) -> tuple[list[str], list[_LaunchCsvGroup]]:
    """Validate and partition one Details CSV without invoking NCU."""
    if (
        type(expected_launch_count) is not int
        or expected_launch_count <= 0
        or type(requested_launch_skip) is not int
        or type(requested_launch_count) is not int
    ):
        raise ValueError(
            "NCU launch-set selectors require strict integers and a positive "
            "expected launch count"
        )
    if requested_launch_skip != 0:
        raise ValueError("NCU launch-set requested launch skip must be 0")
    if requested_launch_count != expected_launch_count + 1:
        raise ValueError(
            "NCU launch-set requested launch count must equal expected count plus one"
        )

    try:
        csv_rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error as exc:
        raise ValueError(f"NCU launch-set CSV is malformed: {exc}") from exc
    if not csv_rows:
        raise ValueError("NCU launch-set CSV has no header")
    header = csv_rows[0]
    missing = [
        name for name in _LAUNCH_SET_IDENTITY_COLUMNS if name not in header
    ]
    if missing:
        raise ValueError(
            f"NCU launch-set identity columns missing: {missing}"
        )
    duplicate = [
        name for name in _LAUNCH_SET_IDENTITY_COLUMNS
        if header.count(name) > 1
    ]
    if duplicate:
        raise ValueError(
            f"NCU launch-set identity columns duplicate: {duplicate}"
        )
    metric_columns = (DEFAULT_METRIC_NAME_COLUMN, DEFAULT_METRIC_VALUE_COLUMN)
    missing_metrics = [name for name in metric_columns if name not in header]
    if missing_metrics:
        raise ValueError(
            f"NCU launch-set metric columns missing: {missing_metrics}"
        )
    duplicate_metrics = [
        name for name in metric_columns if header.count(name) > 1
    ]
    if duplicate_metrics:
        raise ValueError(
            f"NCU launch-set metric columns duplicate: {duplicate_metrics}"
        )
    identity_indexes = {
        name: header.index(name) for name in _LAUNCH_SET_IDENTITY_COLUMNS
    }
    groups_by_id: dict[str, _LaunchCsvGroup] = {}
    ordered_groups: list[_LaunchCsvGroup] = []
    for row_number, row in enumerate(csv_rows[1:], start=2):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(header):
            raise ValueError(
                f"NCU launch-set CSV row {row_number} has {len(row)} columns; "
                f"expected {len(header)}"
            )
        identity = {
            name: row[index].strip()
            for name, index in identity_indexes.items()
        }
        blank = [
            name for name in _LAUNCH_SET_IDENTITY_COLUMNS
            if not identity[name]
        ]
        if blank:
            raise ValueError(
                f"NCU launch-set identity values blank at row {row_number}: {blank}"
            )
        report_id = identity["ID"]
        group = groups_by_id.get(report_id)
        if group is None:
            group = _LaunchCsvGroup(identity=identity, rows=[])
            groups_by_id[report_id] = group
            ordered_groups.append(group)
        elif group.identity != identity:
            drift = [
                name for name in _LAUNCH_SET_IDENTITY_COLUMNS
                if group.identity[name] != identity[name]
            ]
            raise ValueError(
                f"NCU launch-set identity drift for ID {report_id!r}: {drift}"
            )
        normalized_row = list(row)
        for name, index in identity_indexes.items():
            normalized_row[index] = identity[name]
        group.rows.append(normalized_row)

    if len(ordered_groups) != expected_launch_count:
        raise ValueError(
            "NCU launch-set identity cardinality is "
            f"{len(ordered_groups)}; expected {expected_launch_count}"
        )
    return header, ordered_groups


def extract_launch_observations(
    text: str, *, expected_launch_count: int,
    requested_launch_skip: int, requested_launch_count: int,
) -> dict[str, Any]:
    """Return an ordered, unsealed inventory from one launch-set CSV."""
    _header, groups = _split_launch_set_csv(
        text,
        expected_launch_count=expected_launch_count,
        requested_launch_skip=requested_launch_skip,
        requested_launch_count=requested_launch_count,
    )
    return {
        "identity_columns": list(_LAUNCH_SET_IDENTITY_COLUMNS),
        "launch_count": len(groups),
        "launches": [
            {
                "report_launch_ordinal": ordinal,
                "identity": dict(group.identity),
            }
            for ordinal, group in enumerate(groups)
        ],
    }


def extract_launch_observation(text: str) -> dict[str, Any]:
    """Return the one independently imported NCU launch identity."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("NCU CSV has no header")
    missing = [
        name for name in _REQUIRED_OBSERVATION_COLUMNS
        if name not in reader.fieldnames
    ]
    if missing:
        raise ValueError(f"NCU CSV launch identity columns missing: {missing}")
    duplicate = [
        name for name in _OBSERVATION_COLUMNS
        if reader.fieldnames.count(name) > 1
    ]
    if duplicate:
        raise ValueError(f"NCU CSV launch identity columns duplicate: {duplicate}")
    columns = [name for name in _OBSERVATION_COLUMNS if name in reader.fieldnames]
    identities = {
        tuple((row.get(name) or "").strip() for name in columns)
        for row in reader
        if any((row.get(name) or "").strip() for name in columns)
    }
    if len(identities) != 1:
        raise ValueError(
            f"NCU CSV launch identity cardinality is {len(identities)}"
        )
    values = next(iter(identities))
    identity = dict(zip(columns, values, strict=True))
    blank = [name for name in _REQUIRED_OBSERVATION_COLUMNS if not identity[name]]
    if blank:
        raise ValueError(f"NCU CSV launch identity values blank: {blank}")
    return {
        "identity_columns": columns,
        "identity": identity,
    }


def write_launch_observation(
    *, csv_text: str, report_sha256: str, output_path: Path,
    requested_launch_skip: int, requested_launch_count: int,
) -> None:
    observation = extract_launch_observation(csv_text)
    payload = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": requested_launch_skip,
        "requested_launch_count": requested_launch_count,
        "report_sha256": report_sha256,
        **observation,
    }
    encoded_for_seal = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["observation_sha256"] = hashlib.sha256(encoded_for_seal).hexdigest()
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

def validate_launch_observation_document(
    document: Any,
    *,
    requested_launch_skip: int,
    requested_launch_count: int,
    report_sha256: str,
) -> None:
    """Validate a sealed one-launch observation against external bindings."""
    expected_fields = {
        "schema_version",
        "kind",
        "kernel_name_base",
        "requested_launch_skip",
        "requested_launch_count",
        "report_sha256",
        "identity_columns",
        "identity",
        "observation_sha256",
    }
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise ValueError("NCU launch observation fields differ from the closed contract")
    if (
        type(requested_launch_skip) is not int
        or not 0 <= requested_launch_skip <= 2_147_483_647
        or type(requested_launch_count) is not int
        or not 1 <= requested_launch_count <= 2_147_483_647
        or requested_launch_skip + requested_launch_count > 2_147_483_647
    ):
        raise ValueError("NCU launch observation selectors are invalid")
    if (
        not isinstance(report_sha256, str)
        or len(report_sha256) != 64
        or any(character not in "0123456789abcdef" for character in report_sha256)
    ):
        raise ValueError("NCU launch observation report SHA-256 is invalid")
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("kind") != "ncu_details_csv_launch_observation"
        or document.get("kernel_name_base") != "mangled"
        or type(document.get("requested_launch_skip")) is not int
        or document.get("requested_launch_skip") != requested_launch_skip
        or type(document.get("requested_launch_count")) is not int
        or document.get("requested_launch_count") != requested_launch_count
        or document.get("report_sha256") != report_sha256
    ):
        raise ValueError("NCU launch observation authority binding differs")

    columns = document.get("identity_columns")
    identity = document.get("identity")
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        raise ValueError("NCU launch observation identity_columns must be strings")
    canonical_columns = [
        column for column in _OBSERVATION_COLUMNS if column in columns
    ]
    if columns != canonical_columns or not set(_REQUIRED_OBSERVATION_COLUMNS).issubset(
        columns
    ):
        raise ValueError("NCU launch observation identity columns differ")
    if not isinstance(identity, dict) or set(identity) != set(columns):
        raise ValueError("NCU launch observation identity keys differ")
    if any(not isinstance(identity[column], str) for column in columns):
        raise ValueError("NCU launch observation identity values must be strings")
    blank_required = [
        column
        for column in _REQUIRED_OBSERVATION_COLUMNS
        if not identity[column].strip()
    ]
    if blank_required:
        raise ValueError(
            f"NCU launch observation identity values blank: {blank_required}"
        )

    unsigned = dict(document)
    claimed_seal = unsigned.pop("observation_sha256")
    try:
        encoded = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"NCU launch observation is not canonical JSON: {exc}") from exc
    expected_seal = hashlib.sha256(encoded).hexdigest()
    if claimed_seal != expected_seal:
        raise ValueError("NCU launch observation seal differs")


# Columns used to compute a launch-identity key when the canonical
# ``ID`` column is absent. NCU emits ``ID`` for every per-launch row,
# but if a future version drops it the parser falls back to whatever
# subset of these columns is present in the header.
_LAUNCH_IDENTITY_FALLBACK_COLUMNS = (
    "Process ID",
    "Context",
    "Stream",
    "Device",
    "Kernel Name",
    "Kernel Time",
)


# Map each registered ``.pct`` Ratio metric to its NCU Counter pair
# (lookup-hit, lookup-miss). When the caller requests a registered
# ``.pct`` metric, the parser automatically also collects this counter
# pair from the same CSV so an out-of-range ``.pct`` value can be
# replaced with the mathematically-bounded derived ratio
# ``hit / (hit + miss)``. The pair names follow NCU 2026.1's
# ``--query-metrics`` output for sector-basis L2 / L1TEX counters.
_PCT_TO_RAW_COUNTER_PAIR: dict[str, tuple[str, str]] = {
    "lts__t_sector_hit_rate.pct": (
        "lts__t_sectors_lookup_hit.sum",
        "lts__t_sectors_lookup_miss.sum",
    ),
    "l1tex__t_sector_hit_rate.pct": (
        "l1tex__t_sectors_lookup_hit.sum",
        "l1tex__t_sectors_lookup_miss.sum",
    ),
}


def _derive_hit_rate_from_raw_counters(
    hit_str: str, miss_str: str,
) -> float | None:
    """Return ``hit / (hit + miss)`` as a fraction in ``[0, 1]``, or
    ``None`` when the inputs are not parseable as non-negative numbers
    or when the total is zero (no L2 sector lookups occurred).

    NCU emits counter values as non-negative floats (the ``.sum``
    aggregator returns the summed integer count rendered through the
    same ``--print-fp`` path as Ratio metrics). The derived ratio is
    mathematically bounded; a ``None`` return signals the caller to
    fall through to the existing ``ncu_csv_metric_out_of_range``
    typed reason rather than inventing a value.

    NCU formats raw integer counter values with thousands
    separators (e.g., ``2,481.00``) regardless of the
    ``--print-fp`` flag (which targets floating-point metric
    values, not integer counters). Strip ``,`` before
    ``float()`` so the fallback admits comma-bearing counter
    strings instead of returning ``None`` (which would surface
    as ``ncu_csv_metric_out_of_range`` to the caller). The
    same strip discipline applies in the auxiliary-counter and
    primary-metric loops in :func:`parse_raw_page_csv`.
    """
    try:
        hit = float(hit_str.replace(",", ""))
        miss = float(miss_str.replace(",", ""))
    except ValueError:
        return None
    if hit < 0 or miss < 0:
        return None
    total = hit + miss
    if total == 0:
        return None
    derived = hit / total
    if 0.0 <= derived <= 1.0:
        return derived
    return None


@dataclass
class MetricExtractionResult:
    """Outcome of extracting a metric set from raw-page CSV.

    On success ``fraction_values`` carries one float per requested
    metric, all in ``[0, 1]``. On failure ``typed_reason`` carries
    one of the ``ncu_csv_*`` parser reasons or an import-boundary reason:
    ``ncu_import_failed``, ``ncu_import_authority_invalid``,
    ``ncu_import_report_identity_invalid``, ``ncu_import_timeout``, or
    ``ncu_import_output_limit``.

    ``parser_diagnostics`` carries non-fatal observations the parser
    wishes the audit trail to retain. Two diagnostic keys are
    currently emitted:

    - ``ncu_sector_hit_rate_clamped_to_100`` records every metric
      whose raw ``.pct`` value fell in the saturation-rounding window
      ``[100.0, 105.0]`` and was clamped to exactly 1.0.
    - ``ncu_lts_pct_overcount_raw_counter_ratio_used`` records every
      ``.pct`` metric whose raw value exceeded 105.0 but for which the
      registered raw-counter pair (``.sum`` lookup-hit and
      lookup-miss) was present in the CSV and produced a bounded
      derived ratio ``hit / (hit + miss)``. Each record is a
      ``(metric_name, raw_pct_value)`` tuple capturing what the
      ``.pct`` value WAS before the fallback fired.
    """

    ok: bool
    typed_reason: str = ""
    detail: str = ""
    fraction_values: dict[str, float] = field(default_factory=dict)
    parser_diagnostics: dict[str, list[tuple[str, float]]] = field(
        default_factory=dict,
    )


def _select_named_rows(
    text: str,
    wanted: set[str],
    primary_wanted: set[str],
    *,
    name_column: str,
    value_column: str,
) -> "MetricExtractionResult | dict[str, tuple[tuple[str, ...], str]]":
    """Select metric rows from raw-page CSV by metric-name match.

    Shared row-selection core for both the percentage path
    (:func:`parse_raw_page_csv`) and the registry-driven unit-aware path
    (:func:`extract_unit_tagged_metrics`). Returns the ``seen`` mapping
    ``{metric_name: (launch_identity, raw_value_str)}`` on success, or a
    typed-failure ``MetricExtractionResult`` on any of the documented
    structural failures (empty CSV, missing columns, duplicate/cross-launch
    rows, missing primary metrics). Only the PRIMARY metrics are required;
    auxiliary metrics in ``wanted`` (e.g. opportunistic raw-counter pairs)
    are collected when present but their absence is not an error.

    The launch-identity tolerance (admit same-value multi-section rows of a
    single launch; reject distinct launch identities even on identical
    values) is unit-agnostic, so it lives here unchanged.
    """
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return MetricExtractionResult(
            ok=False,
            typed_reason="ncu_csv_empty",
            detail="raw-page CSV produced no rows",
        )

    try:
        name_idx = header.index(name_column)
        value_idx = header.index(value_column)
    except ValueError:
        return MetricExtractionResult(
            ok=False,
            typed_reason="ncu_csv_header_missing_columns",
            detail=(
                f"raw-page CSV header missing '{name_column}' or "
                f"'{value_column}' columns; got header={header!r}"
            ),
        )

    # Build the launch-identity key columns. Prefer the canonical
    # ``ID`` column; fall back to the subset of leading identifier
    # columns present in the header.
    if "ID" in header:
        identity_idxs: tuple[int, ...] = (header.index("ID"),)
        identity_kind = "ID"
    else:
        identity_idxs = tuple(
            header.index(col)
            for col in _LAUNCH_IDENTITY_FALLBACK_COLUMNS
            if col in header
        )
        identity_kind = (
            f"fallback={tuple(c for c in _LAUNCH_IDENTITY_FALLBACK_COLUMNS if c in header)!r}"
        )

    def launch_identity(row: list[str]) -> tuple[str, ...]:
        """Return a stable tuple identifying the launch this row
        belongs to. When the header carries no identity columns at
        all, the tuple is empty — every row collapses to a single
        synthetic launch identity, and the multi-section tolerance
        still applies on same value, while the cross-launch reject
        path is unreachable (which is the safest behavior when the
        CSV truly has no per-launch keys to distinguish)."""
        return tuple(row[idx] if idx < len(row) else "" for idx in identity_idxs)

    seen: dict[str, tuple[tuple[str, ...], str]] = {}
    conflict_counts: dict[str, int] = {}
    cross_launch_counts: dict[str, set[tuple[str, ...]]] = {}
    for row in reader:
        if len(row) <= max(name_idx, value_idx):
            continue
        name = row[name_idx]
        if name not in wanted:
            continue
        value = row[value_idx]
        identity = launch_identity(row)
        if name in seen:
            prev_identity, prev_value = seen[name]
            if identity == prev_identity:
                # Same launch, possibly different section. NCU 2026.1+
                # emits each requested metric in multiple sections of
                # the same launch with identical values; admit. A
                # value disagreement on the same launch is a different
                # defect (NCU producing inconsistent same-launch
                # values) and is rejected as a conflict.
                if value == prev_value:
                    continue
                conflict_counts[name] = conflict_counts.get(name, 1) + 1
                continue
            # Different launch identity — this is the exact
            # launch-filter regression signal R11 reinstated.
            # Reject regardless of whether the values match: even
            # identical values across distinct launches mean
            # ``--launch-skip`` / ``--launch-count 1`` was missing
            # and the import accidentally aggregated multiple
            # launches.
            cross_launch_counts.setdefault(name, {prev_identity}).add(
                identity
            )
            continue
        seen[name] = (identity, value)

    if cross_launch_counts:
        details = ", ".join(
            f"'{n}': {len(ids)} distinct launch identities"
            for n, ids in sorted(cross_launch_counts.items())
        )
        return MetricExtractionResult(
            ok=False,
            typed_reason="ncu_csv_duplicate_metric_rows",
            detail=(
                f"multiple launch identities in details-page CSV — {details} "
                f"(launch-identity key: {identity_kind}). This usually "
                f"means ``--launch-skip`` / ``--launch-count 1`` was "
                f"missing and the import aggregated multiple kernel "
                f"launches; the parser rejects this even when the "
                f"numeric values happen to match."
            ),
        )

    if conflict_counts:
        details = ", ".join(
            f"'{n}': {count} conflicting rows" for n, count in sorted(conflict_counts.items())
        )
        return MetricExtractionResult(
            ok=False,
            typed_reason="ncu_csv_duplicate_metric_rows",
            detail=(
                f"conflicting metric values within a single launch "
                f"identity — {details}. NCU produced disagreeing values "
                f"for the same metric across sections of one launch."
            ),
        )

    # Only PRIMARY metrics are required. Auxiliary metrics (e.g. raw-counter
    # pairs) are opportunistically collected; their absence is not an error.
    missing = primary_wanted - set(seen)
    if missing:
        return MetricExtractionResult(
            ok=False,
            typed_reason="ncu_csv_metric_missing",
            detail=(
                f"raw-page CSV is missing rows for metrics: "
                f"{sorted(missing)}"
            ),
        )
    return seen


def parse_raw_page_csv(
    text: str,
    metric_names: Iterable[str],
    *,
    name_column: str = DEFAULT_METRIC_NAME_COLUMN,
    value_column: str = DEFAULT_METRIC_VALUE_COLUMN,
) -> MetricExtractionResult:
    """Parse the raw-page CSV ``text`` and extract the requested
    metrics. The CSV must have a header row whose names include
    ``name_column`` and ``value_column``; row order is irrelevant
    because rows are selected by metric-name match.

    When the caller requests a ``.pct`` metric registered in
    ``_PCT_TO_RAW_COUNTER_PAIR``, the parser also opportunistically
    collects the matching ``(lookup_hit, lookup_miss)`` counter pair
    from the same CSV. The auxiliary counters are NOT returned in
    ``fraction_values`` (only the originally-requested metrics are);
    they are consumed internally to power the raw-counter ratio
    fallback when ``.pct`` reads above the saturation-rounding
    window. If the counter pair is absent from the CSV the parser
    simply falls through to the existing
    ``ncu_csv_metric_out_of_range`` reject — opportunistic, not
    required."""
    primary_wanted = set(metric_names)
    if not primary_wanted:
        return MetricExtractionResult(
            ok=False,
            typed_reason="ncu_csv_metric_missing",
            detail="caller passed an empty metric_names iterable",
        )

    # Opportunistically extend `wanted` to include the raw-counter
    # pair for every registered ``.pct`` metric the caller requested.
    # These auxiliary counters are extracted from the CSV when present
    # but are not required: their absence does not fail the
    # extraction, and they do not appear in ``fraction_values``.
    auxiliary_counter_pairs: dict[str, tuple[str, str]] = {
        pct: pair
        for pct, pair in _PCT_TO_RAW_COUNTER_PAIR.items()
        if pct in primary_wanted
    }
    wanted = set(primary_wanted)
    for hit_name, miss_name in auxiliary_counter_pairs.values():
        wanted.add(hit_name)
        wanted.add(miss_name)

    selection = _select_named_rows(
        text,
        wanted,
        primary_wanted,
        name_column=name_column,
        value_column=value_column,
    )
    if isinstance(selection, MetricExtractionResult):
        return selection
    seen = selection

    fractions: dict[str, float] = {}
    clamped: list[tuple[str, float]] = []
    derived_from_raw_counter: list[tuple[str, float]] = []
    # Validate auxiliary counter values once up front so a malformed
    # row in the counter pair surfaces as a typed-error rather than a
    # silent fallback decline. Counter metrics have non-negative
    # integer semantics (no [0, 100] ceiling); the only check is
    # non-numeric or strict negative.
    #
    # NCU formats raw integer counter values with thousands
    # separators in the US locale (e.g., ``2,481.00``) regardless
    # of the ``--print-fp`` flag (which targets floating-point
    # metric values, not counters). Strip the comma so the float
    # parse admits; the comma is purely cosmetic in NCU's CSV
    # output and carries no semantic information.
    for hit_name, miss_name in auxiliary_counter_pairs.values():
        for counter_name in (hit_name, miss_name):
            if counter_name not in seen:
                continue
            _, counter_str = seen[counter_name]
            try:
                counter_val = float(counter_str.replace(",", ""))
            except ValueError:
                return MetricExtractionResult(
                    ok=False,
                    typed_reason="ncu_csv_metric_non_numeric",
                    detail=(
                        f"raw-counter metric '{counter_name}' value "
                        f"'{counter_str}' is not numeric"
                    ),
                )
            if counter_val < 0:
                return MetricExtractionResult(
                    ok=False,
                    typed_reason="ncu_csv_metric_out_of_range",
                    detail=(
                        f"raw-counter metric '{counter_name}' value "
                        f"{counter_val} is negative; counters must be "
                        f"non-negative"
                    ),
                )

    for name in primary_wanted:
        _, value_str = seen[name]
        # Same locale-strip discipline as the auxiliary-counter loop
        # above: NCU may emit thousands separators on integer-typed
        # counter rows even when the primary metric is a percentage
        # `.pct` value, and the parser admits both shapes.
        try:
            value = float(value_str.replace(",", ""))
        except ValueError:
            return MetricExtractionResult(
                ok=False,
                typed_reason="ncu_csv_metric_non_numeric",
                detail=(
                    f"metric '{name}' value '{value_str}' is not numeric"
                ),
            )
        # Direct admission window: [0, 100] → fraction = value / 100.
        if 0.0 <= value <= 100.0:
            fractions[name] = value / 100.0
            continue
        # Saturation-rounding window: (100, 105] → 1.0 + existing
        # ``ncu_sector_hit_rate_clamped_to_100`` diagnostic. NCU's
        # counter-sampling quantization produces this slight overshoot
        # under specific replay-mode + cache-control combinations.
        if 100.0 < value <= 105.0:
            clamped.append((name, value))
            fractions[name] = 1.0
            continue
        # Out-of-range positive value (>105): try raw-counter ratio
        # fallback before failing. The fallback applies only to
        # ``.pct`` metrics registered in ``_PCT_TO_RAW_COUNTER_PAIR``
        # whose lookup-hit and lookup-miss counters were also
        # collected from the CSV. The derived ratio is bounded in
        # ``[0, 1]`` by construction; this addresses a sm_120 /
        # Blackwell consumer defect class where ``.pct`` ratio
        # aggregation overcounts beyond the saturation window.
        if value > 105.0 and name in auxiliary_counter_pairs:
            hit_name, miss_name = auxiliary_counter_pairs[name]
            if hit_name in seen and miss_name in seen:
                _, hit_str = seen[hit_name]
                _, miss_str = seen[miss_name]
                derived = _derive_hit_rate_from_raw_counters(
                    hit_str, miss_str,
                )
                if derived is not None:
                    fractions[name] = derived
                    derived_from_raw_counter.append((name, value))
                    continue
        # No fallback available, or fallback returned None (zero
        # total, negative inputs, derived out of [0,1]). Hard reject
        # with the existing typed reason.
        return MetricExtractionResult(
            ok=False,
            typed_reason="ncu_csv_metric_out_of_range",
            detail=(
                f"metric '{name}' value {value} is outside [0, 105] and "
                f"raw-counter ratio fallback is unavailable (counters "
                f"absent, zero, negative, or not registered for this "
                f"metric); this parser expects percentage-typed metrics "
                f"(with a [100, 105] saturation-rounding clamp window "
                f"and an opportunistic raw-counter ratio fallback for "
                f"registered .pct metrics)"
            ),
        )

    diagnostics: dict[str, list[tuple[str, float]]] = {}
    if clamped:
        diagnostics["ncu_sector_hit_rate_clamped_to_100"] = clamped
    if derived_from_raw_counter:
        diagnostics[
            "ncu_lts_pct_overcount_raw_counter_ratio_used"
        ] = derived_from_raw_counter
    return MetricExtractionResult(
        ok=True,
        fraction_values=fractions,
        parser_diagnostics=diagnostics,
    )


def diagnostics_to_json_shape(
    result: MetricExtractionResult,
) -> dict[str, list[dict[str, Any]]]:
    """Translate `result.parser_diagnostics` (in-memory tuple-list
    shape) to the schema-friendly record list shape used by the
    per-launch sidecar's `parser_diagnostics` field.

    The schema record shape is `{metric_label: str, raw_value: number}`;
    field names avoid `import_label.py`'s denylist at lines 48-77 which
    blocks `^metric_name(?:_|$)` to keep raw HW-counter identifiers out
    of label sidecars.
    """
    return {
        diagnostic_name: [
            {"metric_label": metric, "raw_value": raw_value}
            for metric, raw_value in records
        ]
        for diagnostic_name, records in result.parser_diagnostics.items()
    }


def write_diagnostics_json(
    result: MetricExtractionResult, path: Path,
) -> None:
    """Write `diagnostics_to_json_shape(result)` to `path` as a JSON
    object. When the parser had no clamp events the file content is
    `{}` (empty object) — well-formed JSON so the downstream sidecar
    emitter at `tools/local_labels/sidecar_emitter.py` can read it
    and omit the optional `parser_diagnostics` sidecar field."""
    shaped = diagnostics_to_json_shape(result)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(shaped, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _merged_parser_diagnostics_result(
    pct_result: MetricExtractionResult,
    unit_result: "UnitTaggedExtractionResult",
) -> MetricExtractionResult:
    """Return one diagnostics carrier containing both parser paths.

    The CLI writes a single diagnostics JSON for the sidecar emitter,
    so non-fatal diagnostics from the unit-tagged extractor must not be
    dropped when --unit-metric is active.
    """
    diagnostics: dict[str, list[tuple[str, float]]] = {}
    for source in (
        pct_result.parser_diagnostics,
        unit_result.parser_diagnostics,
    ):
        for name, records in source.items():
            diagnostics.setdefault(name, []).extend(records)
    return MetricExtractionResult(
        ok=pct_result.ok,
        typed_reason=pct_result.typed_reason,
        detail=pct_result.detail,
        fraction_values=dict(pct_result.fraction_values),
        parser_diagnostics=diagnostics,
    )


@dataclass
class UnitTaggedExtractionResult:
    """Outcome of registry-driven unit-aware extraction.

    On success ``metrics`` maps each requested canonical NCU metric id to a
    ``{"unit": <unit>, "value": <float>}`` record, the value in its native unit
    (a fraction in ``[0, 1]`` for the ``pct`` family, a non-negative raw
    magnitude for ``count`` / ``ns`` / ``cycles``). ``typed_reason`` carries one
    of the ``ncu_csv_*`` reasons plus ``ncu_csv_metric_unregistered`` (a metric
    not in the registry).
    """

    ok: bool
    typed_reason: str = ""
    detail: str = ""
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    parser_diagnostics: dict[str, list[tuple[str, float]]] = field(
        default_factory=dict,
    )


def apply_registry_transform(
    entry: Any,
    raw_value_str: str,
    *,
    counters: tuple[str, str] | None = None,
) -> tuple[float | None, str, str, tuple[str, str, float] | None]:
    """Apply the registry-declared parser transform to a raw NCU value string.

    The transform is chosen SOLELY from ``entry.parser_transform`` (a
    ``metric_registry.MetricEntry``), so a ``count`` / ``ns`` / ``cycles``
    metric can never be divided by 100 and a ``pct`` metric always travels the
    percentage window. Returns ``(value_or_None, typed_reason, detail,
    diag)`` where ``diag`` is an optional ``(diagnostic_name, metric, raw)``
    triple for a clamp / raw-counter-fallback event.

    - ``percent_to_fraction``: the frozen percentage window — ``[0, 100] ->
      value/100``; ``(100, 105] -> 1.0`` with a clamp diagnostic; ``>105`` ->
      the bounded raw-counter ratio ``hit/(hit+miss)`` when ``counters`` is
      supplied and parses, else ``ncu_csv_metric_out_of_range``.
    - ``identity_nonneg``: strip thousands separators, reject non-numeric /
      negative, return the raw non-negative magnitude unchanged.
    """
    transform = entry.parser_transform
    try:
        value = float(raw_value_str.replace(",", ""))
    except ValueError:
        return (
            None,
            "ncu_csv_metric_non_numeric",
            f"metric '{entry.ncu_metric}' value '{raw_value_str}' is not numeric",
            None,
        )
    if transform == "identity_nonneg":
        if value < 0:
            return (
                None,
                "ncu_csv_metric_out_of_range",
                (
                    f"metric '{entry.ncu_metric}' value {value} is negative; "
                    f"{entry.unit} metrics must be non-negative"
                ),
                None,
            )
        return value, "", "", None
    if transform == "percent_to_fraction":
        if 0.0 <= value <= 100.0:
            return value / 100.0, "", "", None
        if 100.0 < value <= 105.0:
            return (
                1.0,
                "",
                "",
                ("ncu_sector_hit_rate_clamped_to_100", entry.ncu_metric, value),
            )
        if value > 105.0 and counters is not None:
            derived = _derive_hit_rate_from_raw_counters(counters[0], counters[1])
            if derived is not None:
                return (
                    derived,
                    "",
                    "",
                    (
                        "ncu_lts_pct_overcount_raw_counter_ratio_used",
                        entry.ncu_metric,
                        value,
                    ),
                )
        return (
            None,
            "ncu_csv_metric_out_of_range",
            (
                f"percentage metric '{entry.ncu_metric}' value {value} is "
                f"outside [0, 105] and no raw-counter fallback applies"
            ),
            None,
        )
    return (
        None,
        "ncu_csv_metric_unregistered",
        f"metric '{entry.ncu_metric}' has unknown parser_transform {transform!r}",
        None,
    )


def extract_unit_tagged_metrics(
    text: str,
    registry: Any,
    ncu_metrics: Iterable[str],
    *,
    name_column: str = DEFAULT_METRIC_NAME_COLUMN,
    value_column: str = DEFAULT_METRIC_VALUE_COLUMN,
) -> UnitTaggedExtractionResult:
    """Registry-driven, unit-aware extraction of named metrics from raw-page CSV.

    ``registry`` is a ``metric_registry.MetricRegistry``. Each requested
    ``ncu_metric`` is resolved to its registry entry; the entry's
    ``parser_transform`` (never the caller) decides the unit path, so a
    ``count`` / ``ns`` / ``cycles`` value is never pushed through the
    percentage ``/100`` path. An unregistered metric is rejected with
    ``ncu_csv_metric_unregistered``. Row selection / launch-identity dedup is
    the same unit-agnostic core used by :func:`parse_raw_page_csv`.
    """
    requested = list(ncu_metrics)
    primary_wanted = set(requested)
    if not primary_wanted:
        return UnitTaggedExtractionResult(
            ok=False,
            typed_reason="ncu_csv_metric_missing",
            detail="caller passed an empty ncu_metrics iterable",
        )
    entries: dict[str, Any] = {}
    for name in primary_wanted:
        entry = registry.get_by_ncu_metric(name)
        if entry is None:
            return UnitTaggedExtractionResult(
                ok=False,
                typed_reason="ncu_csv_metric_unregistered",
                detail=f"metric '{name}' is not in the metric registry",
            )
        entries[name] = entry

    # Opportunistically collect raw-counter pairs for any registered pct metric
    # so a >105 overshoot can fall back to the bounded ratio.
    auxiliary_counter_pairs: dict[str, tuple[str, str]] = {
        pct: pair
        for pct, pair in _PCT_TO_RAW_COUNTER_PAIR.items()
        if pct in primary_wanted
    }
    wanted = set(primary_wanted)
    for hit_name, miss_name in auxiliary_counter_pairs.values():
        wanted.add(hit_name)
        wanted.add(miss_name)

    selection = _select_named_rows(
        text,
        wanted,
        primary_wanted,
        name_column=name_column,
        value_column=value_column,
    )
    if isinstance(selection, MetricExtractionResult):
        return UnitTaggedExtractionResult(
            ok=False,
            typed_reason=selection.typed_reason,
            detail=selection.detail,
        )
    seen = selection

    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, list[tuple[str, float]]] = {}
    for name in requested:
        entry = entries[name]
        _, raw_value_str = seen[name]
        counters: tuple[str, str] | None = None
        if name in auxiliary_counter_pairs:
            hit_name, miss_name = auxiliary_counter_pairs[name]
            if hit_name in seen and miss_name in seen:
                counters = (seen[hit_name][1], seen[miss_name][1])
        value, reason, detail, diag = apply_registry_transform(
            entry, raw_value_str, counters=counters
        )
        if reason:
            return UnitTaggedExtractionResult(
                ok=False, typed_reason=reason, detail=detail
            )
        metrics[name] = {"unit": entry.unit, "value": value}
        if diag is not None:
            diag_name, diag_metric, diag_raw = diag
            diagnostics.setdefault(diag_name, []).append((diag_metric, diag_raw))
    return UnitTaggedExtractionResult(
        ok=True, metrics=metrics, parser_diagnostics=diagnostics
    )


def _unique_metric_request(
    values: Iterable[str], *, family: str,
) -> list[str]:
    requested = list(values)
    if any(not isinstance(name, str) or not name for name in requested):
        raise ValueError(
            f"NCU launch-set {family} metric requests must be non-empty strings"
        )
    if len(set(requested)) != len(requested):
        raise ValueError(
            f"NCU launch-set {family} metric requests contain duplicates"
        )
    return requested


def _one_launch_csv(header: list[str], rows: list[list[str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def extract_launch_metric_observations(
    text: str,
    registry: Any,
    unit_metric_names: Iterable[str],
    *,
    expected_launch_count: int,
    requested_launch_skip: int,
    requested_launch_count: int,
    percentage_metric_names: Iterable[str] = (
        "lts__t_sector_hit_rate.pct",
        "l1tex__t_sector_hit_rate.pct",
    ),
) -> list[dict[str, Any]]:
    """Split one launch-set CSV and parse each launch independently."""
    percentage_requested = _unique_metric_request(
        percentage_metric_names, family="percentage",
    )
    unit_requested = _unique_metric_request(
        unit_metric_names, family="unit-tagged",
    )
    header, groups = _split_launch_set_csv(
        text,
        expected_launch_count=expected_launch_count,
        requested_launch_skip=requested_launch_skip,
        requested_launch_count=requested_launch_count,
    )
    observations: list[dict[str, Any]] = []
    for ordinal, group in enumerate(groups):
        report_id = group.identity["ID"]
        launch_csv = _one_launch_csv(header, group.rows)
        percentage_result = parse_raw_page_csv(
            launch_csv, percentage_requested,
        )
        if not percentage_result.ok:
            raise ValueError(
                "NCU launch-set percentage metric parse failed for "
                f"ordinal {ordinal}, ID {report_id!r}: "
                f"{percentage_result.typed_reason}: {percentage_result.detail}"
            )
        unit_result = extract_unit_tagged_metrics(
            launch_csv, registry, unit_requested,
        )
        if not unit_result.ok:
            raise ValueError(
                "NCU launch-set unit-tagged metric parse failed for "
                f"ordinal {ordinal}, ID {report_id!r}: "
                f"{unit_result.typed_reason}: {unit_result.detail}"
            )
        merged_diagnostics = _merged_parser_diagnostics_result(
            percentage_result, unit_result,
        )
        merged_diagnostics.parser_diagnostics = {
            name: list(dict.fromkeys(records))
            for name, records in merged_diagnostics.parser_diagnostics.items()
        }
        diagnostics = diagnostics_to_json_shape(merged_diagnostics)
        observations.append({
            "report_launch_ordinal": ordinal,
            "report_id": report_id,
            "fraction_values": percentage_result.fraction_values,
            "unit_tagged_metrics": unit_result.metrics,
            "parser_diagnostics": diagnostics,
        })
    return observations


_NCU_IMPORT_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class ImportedReportCsv:
    """One Details CSV bound to the retained report bytes used for import."""

    text: str
    report_sha256: str
    report_size_bytes: int


def _import_error(
    *, typed_reason: str, detail: str,
) -> tuple[None, MetricExtractionResult]:
    return None, MetricExtractionResult(
        ok=False, typed_reason=typed_reason, detail=detail,
    )


def _executor_import_error(
    exc: ncu_executor.NcuActionExecutorError,
) -> tuple[None, MetricExtractionResult]:
    if exc.code == ncu_executor.NCU_ACTION_INPUT_IDENTITY_INVALID:
        typed_reason = "ncu_import_report_identity_invalid"
    elif exc.code in {
        ncu_executor.NCU_IDENTITY_INPUT_INVALID,
        ncu_executor.NCU_EXECUTABLE_IDENTITY_INVALID,
        ncu_executor.NCU_VERSION_PROBE_FAILED,
        ncu_executor.NCU_VERSION_LINE_MISMATCH,
    }:
        typed_reason = "ncu_import_authority_invalid"
    elif exc.code == ncu_executor.NCU_ACTION_TIMEOUT:
        typed_reason = "ncu_import_timeout"
    elif exc.code == ncu_executor.NCU_ACTION_OUTPUT_LIMIT:
        typed_reason = "ncu_import_output_limit"
    else:
        typed_reason = "ncu_import_failed"
    return _import_error(typed_reason=typed_reason, detail=str(exc))


def _details_csv_payload(output: str) -> str:
    """Return the unique Details CSV after a closed NCU warning preamble."""
    lines = output.splitlines(keepends=True)
    required = {
        DEFAULT_METRIC_NAME_COLUMN,
        DEFAULT_METRIC_VALUE_COLUMN,
        *_REQUIRED_OBSERVATION_COLUMNS,
    }
    candidates: list[int] = []
    for index, line in enumerate(lines):
        try:
            rows = list(csv.reader([line], strict=True))
        except csv.Error:
            continue
        if len(rows) == 1 and required.issubset(rows[0]):
            candidates.append(index)
    if len(candidates) != 1:
        raise ValueError(
            "retained NCU output has no unique Details CSV header"
        )
    header_index = candidates[0]
    unexpected = [
        line.rstrip("\r\n")
        for line in lines[:header_index]
        if line.strip() and not line.startswith("==WARNING==")
    ]
    if unexpected:
        raise ValueError(
            "retained NCU output has unexpected text before Details CSV: "
            f"{unexpected!r}"
        )
    return "".join(lines[header_index:])


def _import_report_csv(
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    report_path: Path,
    *,
    timeout_seconds: float = _NCU_IMPORT_TIMEOUT_SECONDS,
) -> tuple[ImportedReportCsv | None, MetricExtractionResult | None]:
    """Import one retained report through independently authorized NCU.

    The executable and report are both descriptor-bound for the complete action.
    ``ImportedReportCsv.report_sha256`` is the hash observed through the retained
    report capability and is the only report hash downstream launch-observation
    emission may bind.
    """
    try:
        expected = ncu_executor.validate_expected_ncu_identity(
            expected_ncu_identity
        )
    except ncu_executor.NcuActionExecutorError as exc:
        return _executor_import_error(exc)
    try:
        retained_report = immutable.open_retained_regular(report_path)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        return _import_error(
            typed_reason="ncu_import_report_identity_invalid",
            detail=f"could not retain NCU report {report_path}: {exc}",
        )

    imported: ImportedReportCsv | None = None
    error: MetricExtractionResult | None = None
    arguments = [
        "--import",
        retained_report.proc_path,
        "--page",
        "details",
        "--csv",
        # NVBit records mangled function symbols. NCU's Details export
        # demangles by default, so the explicit base is part of the join
        # contract rather than presentation formatting.
        "--print-kernel-base",
        "mangled",
        "--print-metric-name",
        "name",
        "--print-units",
        "base",
        # Force numeric values to floating point in the closed C locale.
        "--print-fp",
    ]
    try:
        action = ncu_executor.execute_ncu_action(
            expected,
            arguments,
            timeout_seconds=timeout_seconds,
            retained_report=retained_report,
        )
        if action.returncode != 0:
            stderr = action.stderr.decode("utf-8", errors="surrogateescape")
            _unused, error = _import_error(
                typed_reason="ncu_import_failed",
                detail=(
                    f"retained NCU --import exited {action.returncode}; "
                    f"stderr={stderr.strip()!r}"
                ),
            )
        else:
            try:
                decoded = action.stdout.decode("utf-8", errors="strict")
                text = _details_csv_payload(decoded)
            except (UnicodeDecodeError, ValueError) as exc:
                _unused, error = _import_error(
                    typed_reason="ncu_import_failed",
                    detail=f"retained NCU --import stdout is not Details CSV: {exc}",
                )
            else:
                imported = ImportedReportCsv(
                    text=text,
                    report_sha256=retained_report.sha256,
                    report_size_bytes=retained_report.size_bytes,
                )
    except ncu_executor.NcuActionExecutorError as exc:
        _unused, error = _executor_import_error(exc)
    finally:
        try:
            retained_report.close()
        except (OSError, immutable.ImmutableArtifactError) as exc:
            if error is None:
                imported = None
                _unused, error = _import_error(
                    typed_reason="ncu_import_report_identity_invalid",
                    detail=f"could not close retained NCU report: {exc}",
                )
    return imported, error


def import_and_parse(
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    report_path: Path,
    metric_names: Iterable[str],
) -> MetricExtractionResult:
    """Import an authorized report and parse requested percentage metrics."""
    imported, error = _import_report_csv(expected_ncu_identity, report_path)
    if error is not None:
        return error
    assert imported is not None
    return parse_raw_page_csv(imported.text, metric_names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import an Nsight Compute report and emit named metrics as "
            "fractions in [0, 1]. Each line of stdout has the form "
            "'<metric_name>\\t<fraction>'. Exit non-zero on any "
            "extraction failure."
        )
    )
    parser.add_argument("--ncu-resolved-path", required=True)
    parser.add_argument("--ncu-executable-sha256", required=True)
    parser.add_argument("--ncu-version-line", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="metric name to extract (e.g. lts__t_sector_hit_rate.pct); repeatable",
    )
    parser.add_argument("--launch-observation-json", type=Path)
    parser.add_argument("--requested-launch-skip", type=int)
    parser.add_argument("--requested-launch-count", type=int)
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=None,
        help=(
            "optional path; when supplied, the parser writes its "
            "non-fatal diagnostics dict (e.g. "
            "`ncu_sector_hit_rate_clamped_to_100` clamp events) to "
            "this path as a JSON object in the schema-friendly "
            "record shape used by the per-launch sidecar's "
            "`parser_diagnostics` field."
        ),
    )
    parser.add_argument(
        "--unit-metric",
        action="append",
        default=None,
        help=(
            "canonical NCU metric id for a non-percentage (count/ns/cycles) "
            "or non-flat-field registry metric to extract through the "
            "registry-driven unit-aware path; repeatable. Requires "
            "--unit-tagged-json. Each metric is resolved against "
            "tools/local_labels/metric_registry.json, so a count/ns/cycles "
            "value is never divided by 100."
        ),
    )
    parser.add_argument(
        "--unit-tagged-json",
        type=Path,
        default=None,
        help=(
            "output path for the unit-tagged-metrics JSON array "
            "([{ncu_metric, unit, value}, ...]) the per-launch sidecar's "
            "optional `unit_tagged_metrics` field consumes. Required when "
            "--unit-metric is given."
        ),
    )
    args = parser.parse_args(argv)
    try:
        expected_ncu_identity = ncu_executor.build_expected_ncu_identity(
            resolved_path=args.ncu_resolved_path,
            executable_sha256=args.ncu_executable_sha256,
            version_line=args.ncu_version_line,
        )
    except ncu_executor.NcuActionExecutorError as exc:
        sys.stderr.write(
            f"parse_ncu_metrics: ncu_import_authority_invalid: {exc}\n"
        )
        return 1

    # Legacy percentage-only path: no --unit-metric requested.
    if not args.unit_metric:
        imported, import_error = _import_report_csv(
            expected_ncu_identity, args.report
        )
        if import_error is not None:
            result = import_error
        else:
            assert imported is not None
            result = parse_raw_page_csv(imported.text, args.metric)
        if not result.ok:
            sys.stderr.write(
                f"parse_ncu_metrics: {result.typed_reason}: {result.detail}\n"
            )
            return 1
        if args.diagnostics_json is not None:
            write_diagnostics_json(result, args.diagnostics_json)
        if args.launch_observation_json is not None:
            if (
                args.requested_launch_skip is None
                or args.requested_launch_count is None
                or args.requested_launch_skip < 0
                or args.requested_launch_count != 1
            ):
                sys.stderr.write(
                    "parse_ncu_metrics: launch observation requires a "
                    "nonnegative skip and count=1\n"
                )
                return 1
            try:
                write_launch_observation(
                    csv_text=imported.text,
                    report_sha256=imported.report_sha256,
                    output_path=args.launch_observation_json,
                    requested_launch_skip=args.requested_launch_skip,
                    requested_launch_count=args.requested_launch_count,
                )
            except (OSError, ValueError) as exc:
                sys.stderr.write(
                    f"parse_ncu_metrics: launch observation failed: {exc}\n"
                )
                return 1
        for name in args.metric:
            sys.stdout.write(f"{name}\t{result.fraction_values[name]}\n")
        return 0

    # Registry-driven unit-tagged path: import the report ONCE and feed the
    # same CSV to both the percentage parser (stdout fractions, unchanged) and
    # the unit-aware extractor (JSON array). cli-as-library composition.
    if args.unit_tagged_json is None:
        sys.stderr.write(
            "parse_ncu_metrics: --unit-metric requires --unit-tagged-json\n"
        )
        return 1
    from metric_registry import load_registry

    imported, error = _import_report_csv(expected_ncu_identity, args.report)
    if error is not None:
        sys.stderr.write(
            f"parse_ncu_metrics: {error.typed_reason}: {error.detail}\n"
        )
        return 1
    assert imported is not None
    csv_text = imported.text
    pct_result = parse_raw_page_csv(csv_text, args.metric)
    if not pct_result.ok:
        sys.stderr.write(
            f"parse_ncu_metrics: {pct_result.typed_reason}: {pct_result.detail}\n"
        )
        return 1
    ut_result = extract_unit_tagged_metrics(
        csv_text, load_registry(), args.unit_metric
    )
    if not ut_result.ok:
        sys.stderr.write(
            f"parse_ncu_metrics: {ut_result.typed_reason}: {ut_result.detail}\n"
        )
        return 1
    if args.diagnostics_json is not None:
        write_diagnostics_json(
            _merged_parser_diagnostics_result(pct_result, ut_result),
            args.diagnostics_json,
        )
    if args.launch_observation_json is not None:
        if (
            args.requested_launch_skip is None
            or args.requested_launch_count is None
            or args.requested_launch_skip < 0
            or args.requested_launch_count != 1
        ):
            sys.stderr.write(
                "parse_ncu_metrics: launch observation requires a "
                "nonnegative skip and count=1\n"
            )
            return 1
        try:
            write_launch_observation(
                csv_text=csv_text,
                report_sha256=imported.report_sha256,
                output_path=args.launch_observation_json,
                requested_launch_skip=args.requested_launch_skip,
                requested_launch_count=args.requested_launch_count,
            )
        except (OSError, ValueError) as exc:
            sys.stderr.write(
                f"parse_ncu_metrics: launch observation failed: {exc}\n"
            )
            return 1
    # Preserve the requested --unit-metric order in the emitted array.
    unit_tagged_array = [
        {
            "ncu_metric": name,
            "unit": ut_result.metrics[name]["unit"],
            "value": ut_result.metrics[name]["value"],
        }
        for name in args.unit_metric
    ]
    with args.unit_tagged_json.open("w", encoding="utf-8") as fh:
        json.dump(unit_tagged_array, fh, indent=2, sort_keys=True)
        fh.write("\n")
    for name in args.metric:
        sys.stdout.write(f"{name}\t{pct_result.fraction_values[name]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
