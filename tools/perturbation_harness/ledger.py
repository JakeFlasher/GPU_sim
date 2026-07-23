"""perturbation_run_ledger producer surface for the perturbation-harness clause.

Authoritative producer-side constants and bundle assembly. Mirrors the
controller-style "producer module" pattern used in
`tools/order_projections/projector.py` and `tools/sharing_locality/classifier.py`:
strict typed-reason set as a frozenset (the parity inventory's
`producer_python_constant` surface), plus a `build_bundle(...)` function
that strict-validates output against the on-disk
`workloads/simbench_bw/schemas/perturbation_run_ledger.schema.json`
before returning.

Loop-scope families: the three
`cache_modifier_ca`/`cache_modifier_cg`/`cache_modifier_cs`
cache-modifier variants (R45 added `.cs` to close the
microbench-suite-coverage criterion's cache-modifier-variants
family) and the seven `working_set_sweep` MiB ladder values.
Other families remain typed-unavailable arms.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

# Producer-side typed-reason set. The parity inventory's
# `producer_python_constant` surface is this constant; the schema enum
# at `$defs/typed_reason` mirrors it surface-for-surface.
PERTURBATION_RUN_LEDGER_TYPED_REASONS: frozenset[str] = frozenset({
    "modifier_optimized_away",
    "correctness_equivalence_failed",
    "arch_feature_unavailable_l2_prefetch_64b",
    "arch_feature_unavailable_l2_prefetch_128b",
    "arch_feature_unavailable_l2_prefetch_256b",
    "arch_feature_unavailable_l2_persisting",
    "arch_feature_unavailable_eviction_buffer",
    "arch_feature_unavailable_carveout",
    "perturbation_compile_failed",
})

# Per-variant family enum the schema's per-record `oneOf` keys on. The
# split (R29 patch per Codex's R28 review) exposes the .ca / .cg
# variants individually so the schema can constrain
# `modifier_optimized_away` to fire only on the cache-modifier
# variants.
PERTURBATION_FAMILIES: frozenset[str] = frozenset({
    "cache_modifier_ca",
    "cache_modifier_cg",
    "cache_modifier_cs",
    "working_set_sweep",
})

CACHE_MODIFIER_FAMILIES: frozenset[str] = frozenset({
    "cache_modifier_ca",
    "cache_modifier_cg",
    "cache_modifier_cs",
})

WORKING_SET_LADDER_MIB: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

TARGET_ARCH = "sm_120-laptop"

# Loop-3 widening: the perturbation harness now supports both local
# arches as peer targets. The set is mirrored into the schema's
# `$defs.target_arch_enum`, the producer helper
# `nvcc_gencode_for_target_arch`, and the controller helper
# `_perturbation_target_arch_for_props`. A new value added to one MUST
# be added to all four (BL-20260507-arch-token-multi-surface-lockstep).
SUPPORTED_TARGET_ARCHES: frozenset[str] = frozenset({
    "sm_86",
    "sm_120-laptop",
})

# Cache-modifier variants admit only when SASS confirms BOTH target source
# loads carry the intended modifier. The dram_stream_stride_mix kernel has
# two target source loads (`in[idx]` and `in[i]`); the working_set_sweep
# family does not perturb the cache operator and admits at >=1.
SASS_EXPECTED_MATCH_COUNT_BY_FAMILY: dict[str, int] = {
    "cache_modifier_ca": 2,
    "cache_modifier_cg": 2,
    "cache_modifier_cs": 2,
    "working_set_sweep": 1,
}

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "workloads" / "simbench_bw" / "schemas"
    / "perturbation_run_ledger.schema.json"
)


def load_schema_validator() -> jsonschema.Draft202012Validator:
    """Load the strict perturbation_run_ledger schema validator."""
    with open(_SCHEMA_PATH) as f:
        schema = json.load(f)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def build_bundle(
    *,
    manifest_pairing_index: dict[str, Any],
    device: dict[str, Any],
    perturbations: list[dict[str, Any]],
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble + strict-validate a perturbation_run_ledger.json bundle.

    Returns the bundle dict on success; raises `jsonschema.ValidationError`
    on schema rejection so callers cannot silently emit a non-conformant
    bundle (the R7/R9 strict-schema lesson family).

    The evidence list is auto-populated with a `perturbation_run_summary`
    note if not provided; callers may supply additional notes.
    """
    if evidence is None:
        evidence = []
    has_summary_note = any(
        isinstance(e, dict)
        and e.get("note_kind") == "perturbation_run_summary"
        for e in evidence
    )
    if not has_summary_note:
        evidence = list(evidence) + [
            _build_summary_note(perturbations)
        ]
    bundle = {
        "schema_version": 1,
        "result_kind": "ok",
        "manifest_pairing_index": manifest_pairing_index,
        "device": device,
        "perturbations": perturbations,
        "evidence": evidence,
    }
    validator = load_schema_validator()
    schema_errors = list(validator.iter_errors(bundle))
    if schema_errors:
        first = schema_errors[0]
        raise jsonschema.ValidationError(
            f"perturbation_run_ledger bundle fails strict schema "
            f"validation at {list(first.absolute_path)}: "
            f"{first.message}"
        )
    return bundle


def _build_summary_note(
    perturbations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the perturbation_run_summary evidence note from the
    authoritative `perturbations[]` list. Mirrors the
    `_recompute_perturbation_run_ledger_section` shape so the
    verifier can recompute counts without trusting the cached note.
    """
    admitted = sum(
        1 for p in perturbations
        if isinstance(p, dict) and p.get("result_kind") == "ok"
    )
    typed_unavail = sum(
        1 for p in perturbations
        if isinstance(p, dict)
        and p.get("result_kind") == "typed_unavailable"
    )
    sass_pass = sum(
        1 for p in perturbations
        if isinstance(p, dict)
        and p.get("result_kind") == "ok"
        and p.get("sass_confirmation_status") == "pass"
    )
    correctness_pass = sum(
        1 for p in perturbations
        if isinstance(p, dict)
        and p.get("result_kind") == "ok"
        and p.get("correctness_equivalence_status") == "pass"
    )
    return {
        "result_kind": "info",
        "note_kind": "perturbation_run_summary",
        "total_perturbation_attempts": len(perturbations),
        "admitted_count": admitted,
        "typed_unavailable_count": typed_unavail,
        "sass_confirmation_pass_count": sass_pass,
        "correctness_equivalence_pass_count": correctness_pass,
    }


def per_family_admitted_count(
    perturbations: list[dict[str, Any]],
) -> dict[str, int]:
    """Per-variant admitted-count map keyed by the schema's per-variant
    `family` enum. Returns 0 for any family with no admitted record."""
    counts = {family: 0 for family in PERTURBATION_FAMILIES}
    for record in perturbations:
        if not isinstance(record, dict):
            continue
        if record.get("result_kind") != "ok":
            continue
        family = record.get("family")
        if family in counts:
            counts[family] += 1
    return counts


def per_typed_reason_count(
    perturbations: list[dict[str, Any]],
) -> dict[str, int]:
    """Per-typed-reason count map keyed by the 9 typed reasons."""
    counts = {reason: 0 for reason in PERTURBATION_RUN_LEDGER_TYPED_REASONS}
    for record in perturbations:
        if not isinstance(record, dict):
            continue
        if record.get("result_kind") != "typed_unavailable":
            continue
        reason = record.get("typed_reason")
        if reason in counts:
            counts[reason] += 1
    return counts


def admitted_records(
    perturbations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter the perturbations list down to admitted records."""
    return [
        p for p in perturbations
        if isinstance(p, dict) and p.get("result_kind") == "ok"
    ]


__all__ = [
    "PERTURBATION_RUN_LEDGER_TYPED_REASONS",
    "PERTURBATION_FAMILIES",
    "CACHE_MODIFIER_FAMILIES",
    "WORKING_SET_LADDER_MIB",
    "TARGET_ARCH",
    "SUPPORTED_TARGET_ARCHES",
    "SASS_EXPECTED_MATCH_COUNT_BY_FAMILY",
    "load_schema_validator",
    "build_bundle",
    "per_family_admitted_count",
    "per_typed_reason_count",
    "admitted_records",
]
