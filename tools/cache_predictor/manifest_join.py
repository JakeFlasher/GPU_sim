"""Manifest-to-trace ingestion helper.

Joins a per-PC ``cache_op_map`` manifest (produced by
``l1_cache_op_classifier``) against a sequence of dynamic-trace
records that carry at least a ``pc`` field and optionally a
``memory_op_class`` tag. For each trace record whose ``pc``
matches an entry in the manifest, the helper attaches the
manifest's ``instruction_cache_op`` + ``l1_eligible`` fields to
the record and verifies the (``memory_op_class``,
``instruction_cache_op``) pair is not an orthogonality defect per
``l1_cache_op_classifier.is_orthogonality_defect``.

The helper is the producer-side bridge between the cuobjdump
classifier output (static, per-binary) and the dynamic trace
records (per-launch, per-warp). The L1 predictor head will
consume the joined records to decide which dynamic accesses
contribute L1 line state.

Records without a matching PC in the manifest carry
``instruction_cache_op=None`` and ``l1_eligible=None``; the L1
predictor head's per-SM walker will filter them as the
"unknown cache-op" arm. Orthogonality defects accumulate in the
join result's ``orthogonality_defects`` list; the count is the
admission gate for the join (a non-empty list fails the join
contract).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, NamedTuple


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


from l1_cache_op_classifier import (  # noqa: E402
    SassProvenanceStatus,
    is_orthogonality_defect,
)


class OrthogonalityDefect(NamedTuple):
    """One (memory_op_class, instruction_cache_op) defect."""

    pc: str
    memory_op_class: str
    instruction_cache_op: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pc": self.pc,
            "memory_op_class": self.memory_op_class,
            "instruction_cache_op": self.instruction_cache_op,
        }


class JoinResult(NamedTuple):
    """Outcome of joining a manifest to a trace record sequence."""

    joined_records: list[dict[str, Any]]
    unmatched_pcs: list[str]
    orthogonality_defects: list[OrthogonalityDefect]
    manifest_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "joined_records": list(self.joined_records),
            "unmatched_pcs": list(self.unmatched_pcs),
            "orthogonality_defects": [
                d.to_dict() for d in self.orthogonality_defects
            ],
            "manifest_status": self.manifest_status,
        }

    @property
    def is_admissible(self) -> bool:
        """True iff the join produced zero orthogonality defects
        AND the manifest's sass_provenance_status was matched."""
        return (
            not self.orthogonality_defects
            and self.manifest_status == SassProvenanceStatus.MATCHED.value
        )


def _normalize_pc(pc: Any) -> str:
    """Normalize a PC value to lowercase ``0x``-prefixed hex.

    Accepts:
      - int (e.g., from the proto's uint32 ``pc`` field)
      - str ``0xN`` (already in classifier wire form)
      - str hex without prefix (e.g., ``0010``)
    """
    if isinstance(pc, int):
        return f"0x{pc:x}"
    if isinstance(pc, str):
        s = pc.lower().strip()
        if s.startswith("0x"):
            hex_part = s[2:].lstrip("0") or "0"
            return f"0x{hex_part}"
        # No prefix; assume hex digits.
        hex_part = s.lstrip("0") or "0"
        return f"0x{hex_part}"
    raise TypeError(
        f"unsupported pc value type {type(pc).__name__}: {pc!r}"
    )


def _index_manifest_entries(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return ``{normalized_pc: manifest_entry_dict}`` for fast lookup."""
    out: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("per_pc_instruction_cache_op", []):
        normalized = _normalize_pc(entry["pc"])
        out[normalized] = entry
    return out


def join_manifest_to_records(
    manifest: dict[str, Any],
    records: Iterable[dict[str, Any]],
) -> JoinResult:
    """Join a classifier manifest to a sequence of trace records.

    Each record is a dict carrying at least a ``pc`` field and
    optionally a ``memory_op_class`` field (e.g., a proto3
    ``instruction`` MessageToDict export, or a synthetic dict
    fixture). The helper attaches ``instruction_cache_op``,
    ``l1_eligible``, and (when matched and an orthogonality
    defect is detected) an ``orthogonality_defect`` flag.

    The original record is not mutated; a shallow copy is
    returned with the attached fields.
    """
    index = _index_manifest_entries(manifest)
    joined: list[dict[str, Any]] = []
    unmatched: list[str] = []
    defects: list[OrthogonalityDefect] = []

    for record in records:
        if "pc" not in record:
            raise ValueError(
                f"trace record missing required 'pc' field: {record!r}"
            )
        pc = _normalize_pc(record["pc"])
        merged: dict[str, Any] = dict(record)
        merged["pc"] = pc

        manifest_entry = index.get(pc)
        if manifest_entry is None:
            merged["instruction_cache_op"] = None
            merged["l1_eligible"] = None
            unmatched.append(pc)
            joined.append(merged)
            continue

        cache_op = manifest_entry["instruction_cache_op"]
        merged["instruction_cache_op"] = cache_op
        merged["l1_eligible"] = manifest_entry["l1_eligible"]

        memory_op_class = record.get("memory_op_class")
        if (
            isinstance(memory_op_class, str)
            and is_orthogonality_defect(
                memory_op_class=memory_op_class,
                instruction_cache_op=cache_op,
            )
        ):
            defect = OrthogonalityDefect(
                pc=pc,
                memory_op_class=memory_op_class,
                instruction_cache_op=cache_op,
            )
            defects.append(defect)
            merged["orthogonality_defect"] = defect.to_dict()

        joined.append(merged)

    return JoinResult(
        joined_records=joined,
        unmatched_pcs=unmatched,
        orthogonality_defects=defects,
        manifest_status=manifest.get(
            "sass_provenance_status",
            SassProvenanceStatus.NO_KERNEL_SECTION.value,
        ),
    )


__all__ = [
    "JoinResult",
    "OrthogonalityDefect",
    "join_manifest_to_records",
]
