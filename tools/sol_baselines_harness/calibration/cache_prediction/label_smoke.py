"""M0 label smoke harness for the calibration subtree.

Drives NCU against three sm_86 microbench cells (broadcast,
coalesced stream, and a capacity-overflow proxy), parses the
CSV output into typed label records, and persists a versioned
smoke artifact carrying the raw CSV SHA-256, binary SHA-256,
metric-version pin, warmup-protocol identifier, and per-cell
label record.

The harness lives under the calibration-only subtree per the
CLAUDE.md Rule 1 carveout: the deployed framework path and
every B200 invocation MUST NOT import this module.

Tests drive the harness via the runner / clock / file-hash
injection seams (`_runner`, `_clock`, `_sha256`) so the
deterministic unit tests do not spawn a real NCU process or
depend on wall-clock or filesystem state. The hardware-gated
real-NCU test runs when `shutil.which("ncu")` returns a path
AND every cell binary is present; otherwise it skips cleanly.
"""

from __future__ import annotations

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sol_baselines_harness.calibration.cache_prediction.label_parser import (
    NcuLabelRecord,
)
from sol_baselines_harness.calibration.cache_prediction.ncu_harvester import (
    _run_checked_ncu,
    harvest_ncu_label,
)


LABEL_SMOKE_SCHEMA_VERSION: str = "1.0"


LABEL_SMOKE_TYPED_REASONS: frozenset[str] = frozenset({
    "label_smoke_artifact_schema_violation",
    "label_smoke_record_invalid",
})


class LabelSmokeError(Exception):
    """Raised when a label-smoke artifact or per-cell record
    fails validation. Carries the typed-reason string for
    downstream typed-unavailable bundle composition.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class LabelSmokeCell:
    """One named microbench cell that the smoke harness will
    drive through NCU. The `binary_path` resolves to the
    pre-built calibration binary; `launch_skip_count` selects
    NCU's `--launch-skip N` argument (must be >= 1 per
    `harvest_ncu_label`).
    """

    cell_name: str
    binary_path: Path
    launch_skip_count: int = 5


@dataclass(frozen=True)
class LabelSmokeRecord:
    """Per-cell smoke record. Carries provenance fields so the
    downstream 7-tuple cell join key + bootstrap CI surfaces
    can compose against this artifact.
    """

    cell_name: str
    raw_csv_sha256: str
    binary_sha256: str
    ncu_metric_version: str
    warmup_protocol: str
    launch_skip_count: int
    label: NcuLabelRecord


@dataclass(frozen=True)
class LabelSmokeArtifact:
    """Versioned smoke artifact. `records` is an immutable
    tuple of per-cell results.
    """

    schema_version: str
    captured_at: str
    host_arch: str
    records: tuple[LabelSmokeRecord, ...]


# Arch-keyed preflight dispatch. When True for a
# given `host_arch`, `run_label_smoke` refuses to invoke NCU
# until the capability probe is parsed + validated. Mirrors the
# dispatch table in `calibration_sweep._PREFLIGHT_REQUIRED_BY_ARCH`.
_PREFLIGHT_REQUIRED_BY_ARCH: dict[str, bool] = {
    "sm_120": True,
}


def _default_clock() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _default_sha256_of_file(path: Path) -> str:
    """Stream-hash a file's bytes through SHA-256. Used for
    `binary_sha256` provenance. Injection seam takes
    precedence in tests.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory byte buffer. Used for
    `raw_csv_sha256` provenance.
    """
    return hashlib.sha256(data).hexdigest()


def run_label_smoke(
    cells: Iterable[LabelSmokeCell],
    *,
    ncu_metric_version: str = "unknown",
    warmup_protocol: str = "ncu_launch_skip_n_count_1",
    host_arch: str = "unknown",
    output_path: Path | None = None,
    probe_path: Path | str | None = None,
    _runner=None,
    _clock=None,
    _sha256_of_file=None,
) -> LabelSmokeArtifact:
    """Drive the 3-cell smoke flow.

    For each cell:
      1. Compute the binary SHA-256 (or use the injected
         `_sha256_of_file` helper).
      2. Call `harvest_ncu_label` with the cell's
         `launch_skip_count`, the metric version, the
         binary SHA-256, and the warmup protocol; the
         runner injection seam is passed through. The
         harvester invokes `parse_ncu_csv` against the
         captured CSV.
      3. The captured CSV bytes are NOT exposed by the
         harvester's public API. To compute the raw CSV
         SHA-256 provenance, the runner wrapper installed
         here captures the stdout bytes alongside the
         subprocess invocation and passes them up.

    Returns a `LabelSmokeArtifact` carrying per-cell records.
    Optionally writes JSON to `output_path` via
    `dump_label_smoke_artifact_to_json`.

    Injection seams (None defaults use the real implementation):
      `_runner`: callable matching `harvest_ncu_label`'s
        `_runner` signature.
      `_clock`: callable returning ISO-8601 UTC timestamp.
      `_sha256_of_file`: callable mapping `Path` to hex
        string (binary digest).
    """
    # the design contract preflight: arch-keyed dispatch — when
    # `_PREFLIGHT_REQUIRED_BY_ARCH` reports the live host's arch as
    # requiring a probe, require + validate the NCU capability probe
    # BEFORE constructing any NCU argv. Data-driven: new arches
    # extend the dispatch table to inherit the gate.
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.label_smoke.run_label_smoke')
    if _PREFLIGHT_REQUIRED_BY_ARCH.get(host_arch, False):
        if probe_path is None:
            from sol_baselines_harness.calibration.cache_prediction.ncu_capability_probe import (
                CapabilityProbeError,
            )
            raise CapabilityProbeError(
                "capability_probe_missing",
                f"host_arch={host_arch!r} requires probe_path; "
                "the A" "C-1 preflight gate refuses to start NCU "
                "runs without a validated capability probe "
                "artifact",
            )
        from sol_baselines_harness.calibration.cache_prediction.ncu_capability_probe import (
            preflight_for_arch,
        )
        preflight_for_arch(host_arch, probe_path)

    clock = _clock if _clock is not None else _default_clock
    file_hash = (
        _sha256_of_file
        if _sha256_of_file is not None
        else _default_sha256_of_file
    )

    records: list[LabelSmokeRecord] = []
    for cell in cells:
        captured_csv = bytearray()

        # Wrap the runner so we can capture the stdout
        # bytes (the raw CSV) for the SHA-256 provenance.
        # When no runner is injected the wrapper delegates
        # to `_run_checked_ncu`, which preserves the
        # typed `NcuHarvesterError("ncu_label_invocation_failed")`
        # path on missing NCU or subprocess failure. The
        # earlier inline `subprocess.run` skipped that
        # typed-error normalization and let raw
        # `FileNotFoundError` escape.
        def _capturing_runner(argv, _inner=_runner):
            inner = _inner if _inner is not None else _run_checked_ncu
            completed = inner(argv)
            stdout = completed.stdout
            captured_csv.extend(stdout.encode("utf-8"))
            return completed

        binary_sha256 = file_hash(cell.binary_path)
        label = harvest_ncu_label(
            cell.binary_path,
            launch_skip_count=cell.launch_skip_count,
            ncu_metric_version=ncu_metric_version,
            binary_sha256=binary_sha256,
            warmup_protocol=warmup_protocol,
            _runner=_capturing_runner,
        )
        raw_csv_sha256 = _sha256_of_bytes(bytes(captured_csv))
        records.append(LabelSmokeRecord(
            cell_name=cell.cell_name,
            raw_csv_sha256=raw_csv_sha256,
            binary_sha256=binary_sha256,
            ncu_metric_version=ncu_metric_version,
            warmup_protocol=warmup_protocol,
            launch_skip_count=cell.launch_skip_count,
            label=label,
        ))

    artifact = LabelSmokeArtifact(
        schema_version=LABEL_SMOKE_SCHEMA_VERSION,
        captured_at=clock(),
        host_arch=host_arch,
        records=tuple(records),
    )

    if output_path is not None:
        dump_label_smoke_artifact_to_json(artifact, output_path)

    return artifact


def validate_label_smoke_artifact(
    artifact: LabelSmokeArtifact,
) -> None:
    """Validate that the artifact carries every required
    field with consistent types + values. Raises
    `LabelSmokeError` with a typed reason on the first
    violation.

    Checks:
      * `schema_version`, `captured_at`, `host_arch` are
        non-empty strings.
      * `records` is a non-empty tuple of `LabelSmokeRecord`.
      * Per-record: every field is non-empty; the embedded
        `NcuLabelRecord` has L1/L2 hit rates in `[0.0, 1.0]`
        and positive sector totals.
    """
    if not isinstance(artifact, LabelSmokeArtifact):
        raise LabelSmokeError(
            "label_smoke_artifact_schema_violation",
            f"expected LabelSmokeArtifact; got "
            f"{type(artifact).__name__}",
        )
    for field_name in ("schema_version", "captured_at", "host_arch"):
        value = getattr(artifact, field_name)
        if not isinstance(value, str) or value == "":
            raise LabelSmokeError(
                "label_smoke_artifact_schema_violation",
                f"artifact field {field_name!r} must be a "
                f"non-empty string; got {value!r}",
            )
    if not isinstance(artifact.records, tuple):
        raise LabelSmokeError(
            "label_smoke_artifact_schema_violation",
            f"artifact.records must be a tuple; got "
            f"{type(artifact.records).__name__}",
        )
    if len(artifact.records) == 0:
        raise LabelSmokeError(
            "label_smoke_artifact_schema_violation",
            "artifact.records is empty; the smoke harness "
            "must produce at least one record",
        )
    for record in artifact.records:
        _validate_label_smoke_record(record)


def _validate_label_smoke_record(record: LabelSmokeRecord) -> None:
    if not isinstance(record, LabelSmokeRecord):
        raise LabelSmokeError(
            "label_smoke_record_invalid",
            f"expected LabelSmokeRecord; got "
            f"{type(record).__name__}",
        )
    for field_name in (
        "cell_name",
        "raw_csv_sha256",
        "binary_sha256",
        "ncu_metric_version",
        "warmup_protocol",
    ):
        value = getattr(record, field_name)
        if not isinstance(value, str) or value == "":
            raise LabelSmokeError(
                "label_smoke_record_invalid",
                f"record field {field_name!r} must be a "
                f"non-empty string; got {value!r}",
            )
    if (
        not isinstance(record.launch_skip_count, int)
        or record.launch_skip_count < 1
    ):
        raise LabelSmokeError(
            "label_smoke_record_invalid",
            f"record launch_skip_count must be a positive "
            f"int; got {record.launch_skip_count!r}",
        )
    if not isinstance(record.label, NcuLabelRecord):
        raise LabelSmokeError(
            "label_smoke_record_invalid",
            f"record label must be NcuLabelRecord; got "
            f"{type(record.label).__name__}",
        )
    label = record.label
    if not (0.0 <= label.l1_global_load_hit_rate <= 1.0):
        raise LabelSmokeError(
            "label_smoke_record_invalid",
            f"record label.l1_global_load_hit_rate must be in "
            f"[0.0, 1.0]; got {label.l1_global_load_hit_rate!r}",
        )
    if not (0.0 <= label.l2_read_hit_rate <= 1.0):
        raise LabelSmokeError(
            "label_smoke_record_invalid",
            f"record label.l2_read_hit_rate must be in "
            f"[0.0, 1.0]; got {label.l2_read_hit_rate!r}",
        )
    if label.l1_global_load_sectors_total <= 0:
        raise LabelSmokeError(
            "label_smoke_record_invalid",
            f"record label.l1_global_load_sectors_total must "
            f"be > 0; got {label.l1_global_load_sectors_total!r}",
        )
    if label.l2_read_sectors_total <= 0:
        raise LabelSmokeError(
            "label_smoke_record_invalid",
            f"record label.l2_read_sectors_total must be > "
            f"0; got {label.l2_read_sectors_total!r}",
        )


def dump_label_smoke_artifact_to_json(
    artifact: LabelSmokeArtifact,
    path: Path,
) -> None:
    """Serialize `artifact` to JSON at `path`. The embedded
    `NcuLabelRecord` and `LabelSmokeRecord` dataclasses are
    converted to dicts via `dataclasses.asdict`.
    """
    payload = {
        "schema_version": artifact.schema_version,
        "captured_at": artifact.captured_at,
        "host_arch": artifact.host_arch,
        "records": [asdict(record) for record in artifact.records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_label_smoke_artifact_from_json(
    path: Path,
) -> LabelSmokeArtifact:
    """Roundtrip companion to
    `dump_label_smoke_artifact_to_json`. Re-hydrates the
    artifact dataclass from JSON. Used by the JSON roundtrip
    test.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = tuple(
        LabelSmokeRecord(
            cell_name=r["cell_name"],
            raw_csv_sha256=r["raw_csv_sha256"],
            binary_sha256=r["binary_sha256"],
            ncu_metric_version=r["ncu_metric_version"],
            warmup_protocol=r["warmup_protocol"],
            launch_skip_count=r["launch_skip_count"],
            label=NcuLabelRecord(**r["label"]),
        )
        for r in payload["records"]
    )
    return LabelSmokeArtifact(
        schema_version=payload["schema_version"],
        captured_at=payload["captured_at"],
        host_arch=payload["host_arch"],
        records=records,
    )


__all__ = [
    "LABEL_SMOKE_SCHEMA_VERSION",
    "LABEL_SMOKE_TYPED_REASONS",
    "LabelSmokeArtifact",
    "LabelSmokeCell",
    "LabelSmokeError",
    "LabelSmokeRecord",
    "dump_label_smoke_artifact_to_json",
    "load_label_smoke_artifact_from_json",
    "run_label_smoke",
    "validate_label_smoke_artifact",
]
