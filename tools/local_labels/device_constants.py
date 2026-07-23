"""Fail-closed loader for the shared arch-keyed device-constant table.

The table (``device_constants.json``, alongside this module) is the single
source of truth for the per-architecture hardware constants the GPU-free
predictor modules consume (sector size, cache capacities, register-file and
shared-memory budgets, occupancy ceilings, peak DRAM bandwidth, and clocks).
Every module resolves a constant through this loader instead of hard-coding a
number, so a value can never enter a computation without a recorded source and
a non-empty provenance string.

Each constant is stored as ``{"value": <number|null>, "source": <enum>,
"provenance": <non-empty string>}``. The loader is FAIL-CLOSED: a lookup that
would return a fabricated or unsourced number raises ``DeviceConstantError``
rather than yielding a silent default. Specifically, ``DeviceConstantTable.get``
rejects:

  (a) an unknown architecture key,
  (b) an unknown constant name,
  (c) a ``null`` or missing value (e.g. the deferred sm_120 constants),
  (d) an empty or missing ``provenance`` string (the fabrication guard),
  (e) a ``source`` outside the allowed enum.

``DeviceConstantTable.available(arch, name)`` is the non-raising probe: it
returns ``True`` only when ``get`` would succeed, so callers can branch on a
deferred (value=null) constant without catching an exception.

The arch-invariant sm_120 constants (32-byte sector, 32-thread warp) carry
their value; every other sm_120 constant is deferred with ``value=null`` and a
provenance string that starts with ``sm_120_deferred_no_live_evidence:``.
Individual sm_86 constants may also be quarantined with ``value=null`` when
their provenance is not strong enough for computation. That satisfies the
cross-arch-retain contract (the table keeps an entry for every constant on every
architecture) without ever claiming an unsupported number.

Public surface:

- ``load_device_constants(path=None) -> DeviceConstantTable`` (cached per path).
- ``DeviceConstantTable`` dataclass with ``.get(arch, name) -> float``,
  ``.available(arch, name) -> bool``, ``.source(arch, name)``,
  ``.provenance(arch, name)``, ``.arches()``, ``.constant_names(arch)``.
- ``DeviceConstant`` dataclass for one resolved cell.
- ``ALLOWED_SOURCES`` constant tuple (the closed source enum).
- ``DeviceConstantError`` raised on any fail-closed condition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONSTANTS_PATH = Path(__file__).resolve().parent / "device_constants.json"

# Closed source enum. Mirrors the on-disk table; a source outside this set is a
# fail-closed rejection.
ALLOWED_SOURCES: tuple[str, ...] = (
    "live_device_query",
    "vendor_whitepaper",
    "cuda_programming_guide",
    "derived",
    "deferred_no_live_evidence",
)

# The required constant keys every consuming module shares. The loader does not
# force a table to carry exactly this set (additive future constants are fine),
# but this is the contract names list the modules resolve against.
REQUIRED_CONSTANT_NAMES: tuple[str, ...] = (
    "sector_size_bytes",
    "warp_size",
    "l1_capacity_bytes",
    "l2_capacity_bytes",
    "num_l2_slices",
    "sm_count",
    "registers_per_sm",
    "max_registers_per_thread",
    "register_allocation_unit",
    "warp_allocation_granularity",
    "shared_mem_per_sm_bytes",
    "shared_mem_allocation_unit_bytes",
    "reserved_shared_memory_per_block_bytes",
    "max_warps_per_sm",
    "max_blocks_per_sm",
    "peak_dram_bytes_per_s",
    "base_sm_clock_mhz",
    "boost_sm_clock_mhz",
    "memory_clock_mhz",
)


class DeviceConstantError(ValueError):
    """Raised when the table is malformed or a lookup would fabricate a value."""


@dataclass(frozen=True)
class DeviceConstant:
    """One resolved device-constant cell with its source and provenance."""

    arch: str
    name: str
    value: float | None
    source: str
    provenance: str

    @property
    def is_resolvable(self) -> bool:
        """True when ``value`` is present (a non-null number)."""
        return self.value is not None


@dataclass(frozen=True)
class DeviceConstantTable:
    """In-memory wrapper over the parsed arch-keyed device-constant table.

    The wrapper holds, per architecture, a name -> DeviceConstant mapping. The
    cells are validated for shape (object with value/source/provenance, source
    in the enum, non-empty provenance) at parse time; the value-presence check
    is deferred to ``get`` so a deferred (null) constant can still be inspected
    via ``available`` / ``source`` / ``provenance`` without raising.
    """

    schema_version: int
    _by_arch: dict[str, dict[str, DeviceConstant]]

    def arches(self) -> tuple[str, ...]:
        return tuple(self._by_arch.keys())

    def constant_names(self, arch: str) -> tuple[str, ...]:
        return tuple(self._arch_map(arch).keys())

    def _arch_map(self, arch: str) -> dict[str, DeviceConstant]:
        try:
            return self._by_arch[arch]
        except KeyError as exc:
            raise DeviceConstantError(
                f"unknown architecture {arch!r}; known arches: "
                f"{', '.join(sorted(self._by_arch))}"
            ) from exc

    def _cell(self, arch: str, name: str) -> DeviceConstant:
        amap = self._arch_map(arch)
        try:
            return amap[name]
        except KeyError as exc:
            raise DeviceConstantError(
                f"unknown device constant {name!r} for arch {arch!r}"
            ) from exc

    def get(self, arch: str, name: str) -> float:
        """Resolve a constant to a float, fail-closed.

        Raises ``DeviceConstantError`` on an unknown arch, an unknown constant,
        a null/missing value, an empty/missing provenance, or a source outside
        the allowed enum. (The shape checks are also enforced at parse time;
        re-checking here keeps ``get`` self-contained for a hand-built table.)
        """
        cell = self._cell(arch, name)
        if cell.source not in ALLOWED_SOURCES:
            raise DeviceConstantError(
                f"device constant {name!r} for arch {arch!r} has source "
                f"{cell.source!r} outside the allowed enum {ALLOWED_SOURCES}"
            )
        if not cell.provenance or not cell.provenance.strip():
            raise DeviceConstantError(
                f"device constant {name!r} for arch {arch!r} has empty "
                f"provenance; refusing to return an unsourced value "
                f"(fabrication guard)"
            )
        if cell.value is None:
            raise DeviceConstantError(
                f"device constant {name!r} for arch {arch!r} is deferred "
                f"(value is null); no live evidence to resolve it. provenance: "
                f"{cell.provenance!r}"
            )
        return float(cell.value)

    def available(self, arch: str, name: str) -> bool:
        """True iff ``get(arch, name)`` would succeed (non-raising probe)."""
        try:
            self.get(arch, name)
        except DeviceConstantError:
            return False
        return True

    def source(self, arch: str, name: str) -> str:
        """Return the recorded source enum for a constant (does not need a
        present value; useful for inspecting a deferred constant)."""
        return self._cell(arch, name).source

    def provenance(self, arch: str, name: str) -> str:
        """Return the provenance string for a constant (does not need a present
        value)."""
        return self._cell(arch, name).provenance

    def cell(self, arch: str, name: str) -> DeviceConstant:
        """Return the full resolved cell (value may be None for a deferred
        constant)."""
        return self._cell(arch, name)


def _parse(raw: Any) -> DeviceConstantTable:
    if not isinstance(raw, dict):
        raise DeviceConstantError("device-constant table root must be a JSON object")
    try:
        schema_version = int(raw["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeviceConstantError(
            f"device-constant table missing/invalid schema_version: {exc}"
        ) from exc
    arches = raw.get("arches")
    if not isinstance(arches, dict) or not arches:
        raise DeviceConstantError(
            "device-constant table 'arches' must be a non-empty object"
        )

    by_arch: dict[str, dict[str, DeviceConstant]] = {}
    for arch, constants in arches.items():
        if not isinstance(constants, dict) or not constants:
            raise DeviceConstantError(
                f"arch {arch!r} must map to a non-empty object of constants"
            )
        cells: dict[str, DeviceConstant] = {}
        for name, spec in constants.items():
            if not isinstance(spec, dict):
                raise DeviceConstantError(
                    f"constant {name!r} for arch {arch!r} must be an object "
                    f"{{value, source, provenance}}"
                )
            if "value" not in spec:
                raise DeviceConstantError(
                    f"constant {name!r} for arch {arch!r} is missing 'value'"
                )
            value = spec["value"]
            if value is not None and not isinstance(value, (int, float)):
                raise DeviceConstantError(
                    f"constant {name!r} for arch {arch!r} has non-numeric "
                    f"value {value!r}"
                )
            if isinstance(value, bool):
                # bool is an int subclass; a boolean is never a device constant.
                raise DeviceConstantError(
                    f"constant {name!r} for arch {arch!r} has boolean value"
                )
            source = spec.get("source")
            if source not in ALLOWED_SOURCES:
                raise DeviceConstantError(
                    f"constant {name!r} for arch {arch!r} has source {source!r} "
                    f"outside the allowed enum {ALLOWED_SOURCES}"
                )
            provenance = spec.get("provenance")
            if not isinstance(provenance, str) or not provenance.strip():
                raise DeviceConstantError(
                    f"constant {name!r} for arch {arch!r} has empty/missing "
                    f"provenance (fabrication guard)"
                )
            cells[name] = DeviceConstant(
                arch=arch,
                name=name,
                value=value,
                source=source,
                provenance=provenance,
            )
        by_arch[arch] = cells
    return DeviceConstantTable(schema_version=schema_version, _by_arch=by_arch)


@lru_cache(maxsize=4)
def _load_cached(resolved_path: str) -> DeviceConstantTable:
    raw = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
    return _parse(raw)


def load_device_constants(path: str | None = None) -> DeviceConstantTable:
    """Load and validate the device-constant table. Cached per resolved path."""
    p = Path(path) if path is not None else _CONSTANTS_PATH
    return _load_cached(str(p.resolve()))


def _format_sm86_summary(table: DeviceConstantTable) -> str:
    lines = [
        f"device_constants schema_version={table.schema_version}",
        f"arches: {', '.join(table.arches())}",
        "sm_86 constants:",
    ]
    for name in table.constant_names("sm_86"):
        cell = table.cell("sm_86", name)
        if cell.is_resolvable:
            lines.append(
                f"  {name} = {table.get('sm_86', name)} "
                f"[{cell.source}]"
            )
        else:
            lines.append(f"  {name} = <deferred> [{cell.source}]")
    resolvable = sum(
        1
        for name in table.constant_names("sm_86")
        if table.available("sm_86", name)
    )
    lines.append(
        f"sm_86: {resolvable}/{len(table.constant_names('sm_86'))} "
        f"constants resolvable"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    _table = load_device_constants()
    print(_format_sm86_summary(_table))
