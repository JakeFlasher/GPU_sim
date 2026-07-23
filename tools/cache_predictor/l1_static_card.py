"""Helper for resolving L1 cache parameters from the published
static-card table at ``tools/cache_predictor/l1_static_card_table.json``.

The L1 predictor head uses this helper as the third provenance
source for ``l1tex_capacity_bytes`` (after manifest and CUDA-
attribute queries). A bundle whose ``l1tex_capacity_source`` is
``static_card_constant`` MUST reference an arch present in the
table; ``resolve_l1_static_card_entry(arch)`` raises
``L1StaticCardTableMiss`` when the arch is missing so the L1
head's fail-close arm can construct a typed_unavailable bundle.

Per the plan's no-hard-coded-capacity discipline ("no L1 capacity constant is
hard-coded into Plan 1 implementation or tests"), this module
does NOT hard-code any capacity values; it loads them from the
on-disk JSON table at load time and validates the file against
its schema (`l1_static_card_table.schema.json`). The validation
fires BEFORE any field coercion, so a malformed `l1_ways` value
(string, fractional, or out-of-bounds integer) fails closed with
a typed exception carrying ``TOKEN="predictor_l1_capacity_invalid"``
rather than silently coercing or raising a raw ``ValueError``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import jsonschema

_HERE = Path(__file__).resolve().parent
_TABLE_PATH = _HERE / "l1_static_card_table.json"
_TABLE_SCHEMA_PATH = _HERE / "l1_static_card_table.schema.json"


class L1StaticCardEntry(NamedTuple):
    """One row from the static-card table.

    ``l1_ways`` is OPTIONAL (None when the arch's set-associative
    geometry has not yet been sourced). When present, the L1
    head can route the cell through the per-(SM, set) walker
    (the DEC-2 model class
    ``l1_model_class=set_associative_refinement``); when None,
    the L1 head falls through to the fully-associative
    baseline."""

    arch: str
    l1tex_capacity_bytes: int
    carveout_max_l1_bytes: int
    carveout_max_shared_bytes: int
    default_l1_bytes: int
    source_reference: str
    l1_ways: int | None = None


class L1StaticCardTableMiss(Exception):
    """Raised when ``resolve_l1_static_card_entry`` is asked for
    an arch not present in the published table. The L1 predictor
    head catches this to construct a fail-close typed_unavailable
    bundle naming the missing arch."""

    TOKEN = "predictor_l1_static_card_table_miss"

    def __init__(self, *, arch: str, available_arches: list[str]):
        super().__init__(
            f"{self.TOKEN}: arch {arch!r} not in the published "
            f"static-card table; available={available_arches!r}. "
            f"To add an arch row, edit "
            f"tools/cache_predictor/l1_static_card_table.json "
            f"with the documented per-SM L1 capacity from NVIDIA "
            f"compute-capability documentation; the schema "
            f"validator at l1_static_card_table.schema.json "
            f"enforces the required fields."
        )
        self.arch = arch
        self.available_arches = list(available_arches)


class L1StaticCardEntryInvalid(Exception):
    """Raised when the static-card table fails schema validation
    or carries a field whose value violates the schema's
    constraints. The L1 predictor head's
    ``_emit_l1_scalar_or_walker_fail_close`` catches this to
    emit a typed_unavailable L1TEX bundle with reason
    ``predictor_l1_capacity_invalid``.

    Distinct from ``L1StaticCardTableMiss`` which is the
    legitimate "geometry absent" path for arches without a
    sourced static-card row (the L1 head falls through to the
    fully-associative baseline in that case). This exception
    fires only when an entry IS present but malformed
    (e.g., ``l1_ways=0``, ``l1_ways="abc"``, ``l1_ways=4.5``),
    which would silently coerce or crash with a raw
    ``ValueError`` before the L1 head could emit a typed
    bundle.
    """

    TOKEN = "predictor_l1_capacity_invalid"

    def __init__(
        self,
        *,
        arch: str | None,
        raw_l1_ways: Any,
        validation_message: str,
    ):
        super().__init__(
            f"{self.TOKEN}: static-card entry for arch "
            f"{arch!r} fails schema validation; raw l1_ways "
            f"= {raw_l1_ways!r}; detail: {validation_message}"
        )
        self.arch = arch
        self.raw_l1_ways = raw_l1_ways
        self.validation_message = validation_message


def _load_table_schema() -> dict:
    """Load the bundled JSON Schema for the static-card table."""
    with _TABLE_SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_l1_static_card_table(
    table_path: Path | None = None,
) -> dict[str, L1StaticCardEntry]:
    """Return ``{arch: L1StaticCardEntry}`` parsed from the
    published table.

    The loader strict-validates the raw JSON against
    ``l1_static_card_table.schema.json`` BEFORE coercing fields,
    so a malformed value (e.g., ``l1_ways=0``,
    ``l1_ways="4"``, ``l1_ways=4.5``) raises
    ``L1StaticCardEntryInvalid`` with
    ``TOKEN="predictor_l1_capacity_invalid"`` rather than
    silently coercing or crashing with a raw ``ValueError``.

    Raises:
      ``FileNotFoundError`` -- the table file is missing on
        disk.
      ``L1StaticCardEntryInvalid`` -- the table fails schema
        validation; carries the offending ``arch`` and raw
        ``l1_ways`` value plus the validation error message.
    """
    path = table_path or _TABLE_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Strict schema validation BEFORE any field coercion. This
    # catches l1_ways=0 (violates minimum=1), l1_ways="4" or
    # l1_ways=4.5 (violates type:integer), and any other
    # schema-level malformation. We surface the offending
    # entry's arch + raw l1_ways in the typed exception so the
    # L1 head's defect evidence is auditable.
    schema = _load_table_schema()
    try:
        jsonschema.validate(raw, schema)
    except jsonschema.ValidationError as exc:
        # Best-effort identification of the offending entry:
        # the validation error's absolute_path typically
        # navigates `entries[<index>][<field>]`; extract the
        # row index to surface the arch.
        offending_arch: str | None = None
        offending_l1_ways: Any = None
        try:
            path_components = list(exc.absolute_path)
            if (
                len(path_components) >= 2
                and path_components[0] == "entries"
            ):
                row_index = path_components[1]
                if isinstance(row_index, int) and 0 <= row_index < len(
                    raw.get("entries", [])
                ):
                    row = raw["entries"][row_index]
                    offending_arch = row.get("arch")
                    offending_l1_ways = row.get("l1_ways")
        except (TypeError, KeyError, IndexError):
            pass
        raise L1StaticCardEntryInvalid(
            arch=offending_arch,
            raw_l1_ways=offending_l1_ways,
            validation_message=exc.message,
        ) from exc

    entries: dict[str, L1StaticCardEntry] = {}
    for entry in raw["entries"]:
        l1_ways_raw = entry.get("l1_ways")
        e = L1StaticCardEntry(
            arch=entry["arch"],
            l1tex_capacity_bytes=int(entry["l1tex_capacity_bytes"]),
            carveout_max_l1_bytes=int(entry["carveout_max_l1_bytes"]),
            carveout_max_shared_bytes=int(
                entry["carveout_max_shared_bytes"]
            ),
            default_l1_bytes=int(entry["default_l1_bytes"]),
            source_reference=entry["source_reference"],
            l1_ways=(
                int(l1_ways_raw)
                if l1_ways_raw is not None else None
            ),
        )
        entries[e.arch] = e
    return entries


def resolve_l1_static_card_entry(
    arch: str, table_path: Path | None = None,
) -> L1StaticCardEntry:
    """Return the static-card entry for ``arch`` or raise
    ``L1StaticCardTableMiss``.

    May also raise ``L1StaticCardEntryInvalid`` (from the
    loader's schema-validation gate) when the table itself
    fails schema validation; the L1 head catches that to emit
    a typed_unavailable bundle distinct from the
    ``L1StaticCardTableMiss`` geometry-absent path.
    """
    entries = load_l1_static_card_table(table_path)
    if arch not in entries:
        raise L1StaticCardTableMiss(
            arch=arch, available_arches=sorted(entries),
        )
    return entries[arch]


__all__ = [
    "L1StaticCardEntry",
    "L1StaticCardEntryInvalid",
    "L1StaticCardTableMiss",
    "load_l1_static_card_table",
    "resolve_l1_static_card_entry",
]
