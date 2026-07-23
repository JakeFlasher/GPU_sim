"""Consumer-side admission for `phase_cycle_ledger_<repeat>_<launch>.json`
sidecars (non-root measurement NVBit phase-injection contract).

The non-root measurement contract's phase-injection negative tests require two typed
diagnostics:

  * `nvbit_phase_injection_overhead_exceeded` — emitted when a
    kernel's A/B Heisenberg-overhead record (`ab_overhead_record`)
    reports an `overhead_pct` greater than the registered admission
    threshold. The kernel is excluded from phase-ledger consumers.
  * `phase_boundary_mapping_unresolved` — emitted when a ledger's
    `phase_boundary_status` is `phase_boundary_mapping_unresolved`
    (the SASS / basic-block analysis could not unambiguously map at
    least one phase boundary). The kernel is excluded from
    phase-ledger consumers.

Both diagnostics fire on the consumer side: they admit a per-launch
ledger record produced by the upcoming consolidated NVBit tracer
extension, validate the phase-injection invariants, and either pass
silently (admissible) or raise the typed exception (inadmissible).

The admission threshold is loaded from `nvbit_phase_injection.toml`
via `load_admission_registry`; per-kernel overrides take precedence
over the registry default.

The schema validation itself happens through the existing
`_FAMILY_SCHEMA_MAP` route (`phase_cycle_ledger_*.json` ->
`phase_cycle_ledger.schema.json`); this consumer assumes the input
record is already schema-valid.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent
ADMISSION_TOML = SUITE_ROOT / "nvbit_phase_injection.toml"


# --- Typed diagnostics ------------------------------------------------------


class NvbitPhaseInjectionOverheadExceeded(Exception):
    TOKEN = "nvbit_phase_injection_overhead_exceeded"

    def __init__(
        self,
        slug: str,
        overhead_pct: float,
        admission_threshold_pct: float,
        record_threshold_pct: float | None = None,
        admitted_flag: bool | None = None,
        defect_kind: str = "overhead_above_threshold",
    ):
        super().__init__(
            f"{self.TOKEN}: slug={slug!r} overhead_pct={overhead_pct} "
            f"admission_threshold_pct={admission_threshold_pct} "
            f"record_threshold_pct={record_threshold_pct} "
            f"admitted_flag={admitted_flag} defect_kind={defect_kind!r}; "
            "the kernel is excluded from phase-ledger consumers per the "
            "non-root measurement contract."
        )
        self.slug = slug
        self.overhead_pct = overhead_pct
        self.admission_threshold_pct = admission_threshold_pct
        self.record_threshold_pct = record_threshold_pct
        self.admitted_flag = admitted_flag
        self.defect_kind = defect_kind

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "slug": self.slug,
            "overhead_pct": self.overhead_pct,
            "admission_threshold_pct": self.admission_threshold_pct,
            "record_threshold_pct": self.record_threshold_pct,
            "admitted_flag": self.admitted_flag,
            "defect_kind": self.defect_kind,
        }


class PhaseBoundaryMappingUnresolved(Exception):
    TOKEN = "phase_boundary_mapping_unresolved"

    def __init__(
        self,
        slug: str,
        unresolved_phase_kinds: tuple[str, ...] = (),
    ):
        super().__init__(
            f"{self.TOKEN}: slug={slug!r} could not resolve at least one "
            f"phase boundary; unresolved_phase_kinds={list(unresolved_phase_kinds)}. "
            "The kernel is excluded from phase-ledger consumers per the "
            "non-root measurement contract."
        )
        self.slug = slug
        self.unresolved_phase_kinds = unresolved_phase_kinds

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "slug": self.slug,
            "unresolved_phase_kinds": list(self.unresolved_phase_kinds),
        }


# --- Admission registry -----------------------------------------------------


@dataclass(frozen=True)
class PhaseInjectionAdmission:
    """Loaded `nvbit_phase_injection.toml` view: the default admission
    threshold + the optional per-kernel override map."""

    drift_tolerance_ns_default: int
    default_admission_threshold_pct: float
    per_kernel_admission_threshold_pct: dict[str, float]


def load_admission_registry(
    path: Path = ADMISSION_TOML,
) -> PhaseInjectionAdmission:
    """Load the phase-injection / dual-run admission registry from
    `nvbit_phase_injection.toml`. Per-kernel overrides land in
    `[per_kernel.<slug>]`; missing slugs fall back to the
    `[defaults]` section."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    defaults = data["defaults"]
    per_kernel_raw = data.get("per_kernel", {})
    per_kernel: dict[str, float] = {}
    for slug, override in per_kernel_raw.items():
        if "admission_threshold_pct" in override:
            per_kernel[slug] = float(override["admission_threshold_pct"])
    return PhaseInjectionAdmission(
        drift_tolerance_ns_default=int(defaults["drift_tolerance_ns"]),
        default_admission_threshold_pct=float(
            defaults["default_admission_threshold_pct"]
        ),
        per_kernel_admission_threshold_pct=per_kernel,
    )


def admission_threshold_for_slug(
    admission: PhaseInjectionAdmission, slug: str
) -> float:
    return admission.per_kernel_admission_threshold_pct.get(
        slug, admission.default_admission_threshold_pct
    )


# --- Per-record admission ---------------------------------------------------


def admit_phase_cycle_ledger(
    record: dict[str, Any],
    *,
    admission_threshold_pct: float | None = None,
    admission: PhaseInjectionAdmission | None = None,
) -> None:
    """Validate the phase-injection invariants on a single
    phase-cycle-ledger record. Raises the typed diagnostics on
    defect; passes silently on admissible records.

    The threshold is resolved in the following order:
      1. The explicit `admission_threshold_pct` keyword if provided.
      2. The per-kernel override in `admission` if provided.
      3. The `default_admission_threshold_pct` in `admission`.
      4. The committed registry's default if `admission` is None.

    The schema validation itself happens through the existing
    `_FAMILY_SCHEMA_MAP` route; this function assumes the record is
    already schema-valid."""
    slug = str(record.get("slug", ""))
    if admission_threshold_pct is None:
        if admission is None:
            admission = load_admission_registry()
        admission_threshold_pct = admission_threshold_for_slug(admission, slug)

    # phase-boundary status check: any record whose
    # phase_boundary_status is the unresolved enum value must emit
    # the typed diagnostic. The schema admits absence of the field
    # (defaults to "resolved" semantically); only the explicit
    # unresolved enum triggers the diagnostic.
    boundary_status = record.get("phase_boundary_status", "resolved")
    if boundary_status == "phase_boundary_mapping_unresolved":
        unresolved_kinds: list[str] = []
        for phase_def in record.get("phase_definitions", []):
            if phase_def.get("source_location_hint", "").startswith(
                "unresolved:"
            ):
                unresolved_kinds.append(phase_def.get("phase_kind", "unknown"))
        raise PhaseBoundaryMappingUnresolved(
            slug=slug, unresolved_phase_kinds=tuple(unresolved_kinds)
        )

    # A/B overhead-bound check. The schema requires three invariants
    # on `ab_overhead_record`:
    #   (a) `admission_threshold_pct` MUST equal the resolved registry/
    #       override threshold the producer applied (otherwise the
    #       producer was working against a stale or wrong threshold).
    #   (b) `admitted` MUST equal `overhead_pct <= admission_threshold_pct`
    #       per the schema description: "True iff overhead_pct is at or
    #       below admission_threshold_pct".
    #   (c) The kernel is admitted only when both (a) and (b) hold AND
    #       `overhead_pct <= resolved_threshold`.
    # All three checks are load-bearing: the consumer rejects a record
    # whose self-reported `admitted` flag contradicts its own field
    # values, OR whose record threshold disagrees with the resolved
    # threshold.
    ab = record.get("ab_overhead_record", {})
    overhead_pct = float(ab.get("overhead_pct", 0.0))
    record_threshold = float(ab.get("admission_threshold_pct", 0.0))
    admitted_flag = bool(ab.get("admitted", True))
    if record_threshold != admission_threshold_pct:
        raise NvbitPhaseInjectionOverheadExceeded(
            slug=slug,
            overhead_pct=overhead_pct,
            admission_threshold_pct=admission_threshold_pct,
            record_threshold_pct=record_threshold,
            admitted_flag=admitted_flag,
            defect_kind="record_threshold_disagrees_with_resolved",
        )
    expected_admitted = overhead_pct <= record_threshold
    if admitted_flag != expected_admitted:
        raise NvbitPhaseInjectionOverheadExceeded(
            slug=slug,
            overhead_pct=overhead_pct,
            admission_threshold_pct=admission_threshold_pct,
            record_threshold_pct=record_threshold,
            admitted_flag=admitted_flag,
            defect_kind="admitted_flag_disagrees_with_overhead_vs_threshold",
        )
    if not admitted_flag:
        raise NvbitPhaseInjectionOverheadExceeded(
            slug=slug,
            overhead_pct=overhead_pct,
            admission_threshold_pct=admission_threshold_pct,
            record_threshold_pct=record_threshold,
            admitted_flag=admitted_flag,
            defect_kind="overhead_above_threshold",
        )
