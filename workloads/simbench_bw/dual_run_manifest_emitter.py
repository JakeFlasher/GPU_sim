"""Offline replay assembler for historical `dual_run_manifest.json` evidence.

Reads the archived sidecar shapes from the retired non-root measurement
dual-run workflow. It does not invoke or stand in for a live producer:

  * CUPTI side: `cupti_activity_<repeat>.json` per-launch sidecars
    (one file per kernel launch). Each file is a top-level
    historical CUPTI Activity v1 object carrying
    `slug`, `shape_point`, `repeat_index`, `launch_index`,
    `kernel_signature_sha256`, `binary_sha256`,
    `preflight_manifest_sha256`, `start_ns`, `end_ns`, etc.
  * NVBit side: `phase_cycle_ledger_<repeat>_<launch>.json`
    per-launch sidecars carrying `slug`, `shape_point`,
    `repeat_index`, `launch_index`, `kernel_signature_sha256`,
    `binary_sha256`, `nvbit_tracer_build_hash`, etc.

Per-launch NVML state is derived from the sibling
`nvml_covariate_<repeat>.json` sidecars (per the non-root measurement contract's
NVML-covariate work) by aligning
each launch's `(start_ns, end_ns)` window with the NVML sample
whose `timestamp_ns` falls in the window. If no NVML sidecar exists
on the NVBit side, the emitter raises
`DualRunNvbitNvmlMissing` rather than emit a manifest with
placeholder NVML state.

The assembled manifest is schema-validated against
`dual_run_manifest.schema.json` BEFORE writing the tmp file. If
schema-check fails, the emitter raises
`DualRunManifestSchemaInvalid` and no `<output>.tmp` lands on disk.

Empty input directories raise `DualRunManifestEmptyInputDir`
immediately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dual_run_manifest_consumer as _consumer  # type: ignore[import-not-found]
import validate as _validate  # type: ignore[import-not-found]


_SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"
_HISTORICAL_CUPTI_SCHEMA = "historical/cupti_activity_sidecar.v1.schema.json"


# --- Typed diagnostics ------------------------------------------------------


class DualRunPreflightDisagreement(Exception):
    TOKEN = "dual_run_preflight_disagreement"

    def __init__(self, cupti_preflight_sha256: str, nvbit_preflight_sha256: str):
        super().__init__(
            f"{self.TOKEN}: cupti_preflight_sha256={cupti_preflight_sha256!r} "
            f"!= nvbit_preflight_sha256={nvbit_preflight_sha256!r}; the two "
            "runs anchored against different preflight manifests; the dual-run "
            "manifest cannot be emitted without an amendment."
        )
        self.cupti_preflight_sha256 = cupti_preflight_sha256
        self.nvbit_preflight_sha256 = nvbit_preflight_sha256

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "cupti_preflight_sha256": self.cupti_preflight_sha256,
            "nvbit_preflight_sha256": self.nvbit_preflight_sha256,
        }


class DualRunManifestUnmatchedLaunches(Exception):
    TOKEN = "dual_run_manifest_unmatched_launches"

    def __init__(
        self,
        unmatched_cupti: tuple[dict[str, Any], ...] = (),
        unmatched_nvbit: tuple[dict[str, Any], ...] = (),
    ):
        super().__init__(
            f"{self.TOKEN}: unmatched CUPTI launches: "
            f"{list(unmatched_cupti)}; unmatched NVBit launches: "
            f"{list(unmatched_nvbit)}. Every launch must pair across "
            "the two runs by `(slug, shape_point, repeat_index, "
            "launch_index, kernel_signature_sha256)`."
        )
        self.unmatched_cupti = unmatched_cupti
        self.unmatched_nvbit = unmatched_nvbit

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "unmatched_cupti": list(self.unmatched_cupti),
            "unmatched_nvbit": list(self.unmatched_nvbit),
        }


class DualRunManifestEmptyInputDir(Exception):
    TOKEN = "dual_run_manifest_empty_input_dir"

    def __init__(self, run_dir: Path, run_kind: str):
        super().__init__(
            f"{self.TOKEN}: {run_kind} run dir {str(run_dir)!r} contains no "
            f"per-launch sidecars; cannot emit a dual-run manifest from "
            "an empty input directory. Supply the archived sidecars first."
        )
        self.run_dir = run_dir
        self.run_kind = run_kind

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "run_dir": str(self.run_dir),
            "run_kind": self.run_kind,
        }


class DualRunNvbitNvmlMissing(Exception):
    TOKEN = "dual_run_nvbit_nvml_missing"

    def __init__(self, nvbit_run_dir: Path, missing_repeat_indices: tuple[int, ...]):
        super().__init__(
            f"{self.TOKEN}: NVBit run dir {str(nvbit_run_dir)!r} has no "
            f"nvml_covariate_<repeat>.json sidecar(s) for repeat "
            f"indices {list(missing_repeat_indices)}. The dual-run-manifest "
            "schema requires per-launch NVML state on both sides; the NVBit "
            "run must enable NVML covariate polling before the manifest can "
            "be emitted."
        )
        self.nvbit_run_dir = nvbit_run_dir
        self.missing_repeat_indices = missing_repeat_indices

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "nvbit_run_dir": str(self.nvbit_run_dir),
            "missing_repeat_indices": list(self.missing_repeat_indices),
        }


class DualRunManifestSchemaInvalid(Exception):
    TOKEN = "dual_run_manifest_schema_invalid"

    def __init__(self, schema_path: str, validation_error: str):
        super().__init__(
            f"{self.TOKEN}: assembled manifest failed schema-check against "
            f"{schema_path!r}: {validation_error}. The emitter refuses to "
            "write a schema-invalid manifest."
        )
        self.schema_path = schema_path
        self.validation_error = validation_error

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "schema_path": self.schema_path,
            "validation_error": self.validation_error,
        }


class DualRunInputSchemaInvalid(Exception):
    TOKEN = "dual_run_input_schema_invalid"

    def __init__(
        self,
        sidecar_path: str,
        schema_filename: str,
        validation_error: str,
    ):
        super().__init__(
            f"{self.TOKEN}: input sidecar {sidecar_path!r} failed schema-check "
            f"against {schema_filename!r}: {validation_error}. The emitter "
            "refuses to consume a schema-invalid input."
        )
        self.sidecar_path = sidecar_path
        self.schema_filename = schema_filename
        self.validation_error = validation_error

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "sidecar_path": self.sidecar_path,
            "schema_filename": self.schema_filename,
            "validation_error": self.validation_error,
        }


class DualRunInputDuplicateLaunchKey(Exception):
    TOKEN = "dual_run_input_duplicate_launch_key"

    def __init__(
        self,
        run_kind: str,
        launch_match_key: dict[str, Any],
        first_path: str,
        second_path: str,
    ):
        super().__init__(
            f"{self.TOKEN}: {run_kind} run carries two sidecars with the same "
            f"launch_match_key={launch_match_key!r}; first at {first_path!r}, "
            f"second at {second_path!r}. The emitter refuses to silently "
            "overwrite duplicates."
        )
        self.run_kind = run_kind
        self.launch_match_key = launch_match_key
        self.first_path = first_path
        self.second_path = second_path

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "run_kind": self.run_kind,
            "launch_match_key": self.launch_match_key,
            "first_path": self.first_path,
            "second_path": self.second_path,
        }


class DualRunNvmlAlignmentMissing(Exception):
    TOKEN = "dual_run_nvml_alignment_missing"

    def __init__(
        self,
        launch_match_key: dict[str, Any],
        side: str,
    ):
        super().__init__(
            f"{self.TOKEN}: launch_match_key={launch_match_key!r} has no "
            f"in-window NVML sample on the {side!r} side. The non-root measurement "
            "dual-run-manifest contract requires per-launch NVML state on "
            "both sides; missing alignment cannot be silently dropped."
        )
        self.launch_match_key = launch_match_key
        self.side = side

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "launch_match_key": self.launch_match_key,
            "side": self.side,
        }


# --- Per-run metadata harvest -----------------------------------------------


@dataclass
class _LaunchRecord:
    """Per-launch record harvested from a CUPTI or NVBit sidecar.
    Both run kinds populate the same fields; the NVML state is
    aligned later."""

    slug: str
    shape_point: str
    repeat_index: int
    launch_index: int
    kernel_signature_sha256: str
    binary_sha256: str
    preflight_manifest_sha256: str
    start_ns: int
    end_ns: int


@dataclass
class DualRunMetadata:
    run_id: str
    run_kind: str  # "cupti" or "nvbit"
    binary_sha256: str
    preflight_manifest_sha256: str
    launches_by_key: dict[tuple, _LaunchRecord] = field(default_factory=dict)
    # Per-repeat NVML covariate samples; key is repeat_index.
    nvml_samples_by_repeat: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    # Repeats that the CUPTI sidecars covered but for which no NVML
    # sidecar was found.
    repeats_without_nvml: set[int] = field(default_factory=set)


def _key_from_record(rec: _LaunchRecord) -> tuple:
    return (
        rec.slug,
        rec.shape_point,
        rec.repeat_index,
        rec.launch_index,
        rec.kernel_signature_sha256,
    )


def _stable_run_id(run_kind: str, run_dir: Path, binary_sha256: str) -> str:
    h = hashlib.sha256()
    h.update(run_kind.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(run_dir.resolve()).encode("utf-8"))
    h.update(b"\x00")
    h.update(binary_sha256.encode("utf-8"))
    digest = h.hexdigest()[:32]
    return f"{run_kind}_{digest}"


# --- Sidecar parsing --------------------------------------------------------


def _load_schema(schema_filename: str) -> tuple[dict[str, Any], Path]:
    schema_path = _SCHEMAS_DIR / schema_filename
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return schema, schema_path


def _validate_input_sidecar(
    record: dict[str, Any], schema_filename: str, sidecar_path: Path
) -> None:
    schema, schema_path = _load_schema(schema_filename)
    try:
        _validate.validate(record, schema, root_path=schema_path)
    except _validate.ValidationError as e:
        raise DualRunInputSchemaInvalid(
            sidecar_path=str(sidecar_path),
            schema_filename=schema_filename,
            validation_error=str(e),
        ) from e


def _read_cupti_sidecars(run_dir: Path) -> list[tuple[Path, _LaunchRecord]]:
    out: list[tuple[Path, _LaunchRecord]] = []
    for path in sorted(run_dir.glob("cupti_activity_*.json")):
        if not _validate.is_cupti_activity_sidecar_filename(path.name):
            continue
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        _validate_input_sidecar(
            sidecar, _HISTORICAL_CUPTI_SCHEMA, path
        )
        out.append(
            (
                path,
                _LaunchRecord(
                    slug=str(sidecar["slug"]),
                    shape_point=str(sidecar["shape_point"]),
                    repeat_index=int(sidecar["repeat_index"]),
                    launch_index=int(sidecar["launch_index"]),
                    kernel_signature_sha256=str(sidecar["kernel_signature_sha256"]),
                    binary_sha256=str(sidecar["binary_sha256"]),
                    preflight_manifest_sha256=str(
                        sidecar["preflight_manifest_sha256"]
                    ),
                    start_ns=int(sidecar["start_ns"]),
                    end_ns=int(sidecar["end_ns"]),
                ),
            )
        )
    return out


def _read_nvbit_ledgers(run_dir: Path) -> list[tuple[Path, _LaunchRecord]]:
    """Read NVBit-side phase-cycle ledgers, schema-validated.

    The phase-cycle-ledger schema is a closed object that does NOT
    include `preflight_manifest_sha256` per its non-root measurement-frozen
    schema; we
    leave the ledger record's `preflight_manifest_sha256` empty here
    and harvest the NVBit-side preflight hash from the NVBit-run
    `nvml_covariate_<repeat>.json` sidecar instead (which DOES carry
    that field per `nvml_covariate.schema.json`)."""
    out: list[tuple[Path, _LaunchRecord]] = []
    for path in sorted(run_dir.glob("phase_cycle_ledger_*.json")):
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        _validate_input_sidecar(
            sidecar, "phase_cycle_ledger.schema.json", path
        )
        out.append(
            (
                path,
                _LaunchRecord(
                    slug=str(sidecar["slug"]),
                    shape_point=str(sidecar["shape_point"]),
                    repeat_index=int(sidecar["repeat_index"]),
                    launch_index=int(sidecar["launch_index"]),
                    kernel_signature_sha256=str(sidecar["kernel_signature_sha256"]),
                    binary_sha256=str(sidecar["binary_sha256"]),
                    # phase-ledger schema has no preflight field; the
                    # NVBit-side preflight hash is harvested from the
                    # NVBit-run NVML covariate sidecar in
                    # collect_run_metadata.
                    preflight_manifest_sha256="",
                    start_ns=0,
                    end_ns=0,
                ),
            )
        )
    return out


def _read_nvml_covariate_sidecars(
    run_dir: Path,
) -> tuple[
    dict[int, list[dict[str, Any]]],  # samples by repeat
    dict[int, str],  # preflight_manifest_sha256 by repeat
    dict[int, str],  # binary_sha256 by repeat
]:
    by_repeat: dict[int, list[dict[str, Any]]] = {}
    preflight_by_repeat: dict[int, str] = {}
    binary_by_repeat: dict[int, str] = {}
    for path in sorted(run_dir.glob("nvml_covariate_*.json")):
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        _validate_input_sidecar(sidecar, "nvml_covariate.schema.json", path)
        repeat_index = int(sidecar["repeat_index"])
        by_repeat[repeat_index] = list(sidecar["samples"])
        preflight_by_repeat[repeat_index] = str(sidecar["preflight_manifest_sha256"])
        binary_by_repeat[repeat_index] = str(sidecar["binary_sha256"])
    return by_repeat, preflight_by_repeat, binary_by_repeat


def collect_run_metadata(run_dir: Path, run_kind: str) -> DualRunMetadata:
    """Read a per-run measurement directory and harvest the
    binary_sha256, preflight_manifest_sha256, per-launch records, and
    per-repeat NVML covariate samples.

    Each input sidecar is schema-validated before being indexed.
    Duplicate per-launch natural keys raise
    `DualRunInputDuplicateLaunchKey` instead of being silently
    overwritten.

    For NVBit runs the `preflight_manifest_sha256` is NOT read from
    the phase ledger (which has no such field per the schema); it is
    harvested from the NVBit-run `nvml_covariate_<repeat>.json`
    sidecar, which DOES carry the field per its schema."""
    if run_kind == "cupti":
        records = _read_cupti_sidecars(run_dir)
    elif run_kind == "nvbit":
        records = _read_nvbit_ledgers(run_dir)
    else:
        raise ValueError(f"unknown run_kind: {run_kind!r}")

    if not records:
        raise DualRunManifestEmptyInputDir(run_dir=run_dir, run_kind=run_kind)

    nvml_samples, nvml_preflight_by_repeat, _ = (
        _read_nvml_covariate_sidecars(run_dir)
    )

    launches_by_key: dict[tuple, _LaunchRecord] = {}
    seen_paths: dict[tuple, Path] = {}
    for path, rec in records:
        key = _key_from_record(rec)
        if key in launches_by_key:
            raise DualRunInputDuplicateLaunchKey(
                run_kind=run_kind,
                launch_match_key={
                    "slug": key[0],
                    "shape_point": key[1],
                    "repeat_index": key[2],
                    "launch_index": key[3],
                    "kernel_signature_sha256": key[4],
                },
                first_path=str(seen_paths[key]),
                second_path=str(path),
            )
        launches_by_key[key] = rec
        seen_paths[key] = path

    # Run-level binary + preflight: prefer the per-launch sidecar
    # (which carries them per schema) for CUPTI; for NVBit, fall back
    # to the NVML covariate sidecar because the phase ledger doesn't
    # carry preflight_manifest_sha256.
    first_record = next(iter(launches_by_key.values()))
    binary_sha256 = first_record.binary_sha256
    if run_kind == "cupti":
        preflight_sha256 = first_record.preflight_manifest_sha256
    else:
        # NVBit: phase ledger has no preflight; harvest from the
        # NVBit-run NVML covariate sidecar (per nvml_covariate schema).
        if nvml_preflight_by_repeat:
            preflight_sha256 = next(iter(nvml_preflight_by_repeat.values()))
        else:
            preflight_sha256 = ""

    repeats_seen = {r.repeat_index for r in launches_by_key.values()}
    repeats_without_nvml = repeats_seen - set(nvml_samples.keys())

    return DualRunMetadata(
        run_id=_stable_run_id(run_kind, run_dir, binary_sha256),
        run_kind=run_kind,
        binary_sha256=binary_sha256,
        preflight_manifest_sha256=preflight_sha256,
        launches_by_key=launches_by_key,
        nvml_samples_by_repeat=nvml_samples,
        repeats_without_nvml=repeats_without_nvml,
    )


# --- Per-launch NVML alignment ----------------------------------------------


def _aggregate_nvml_state_at_launch(
    cupti_record: _LaunchRecord,
    nvml_samples_for_repeat: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the NVML sample whose `timestamp_ns` falls within the
    CUPTI launch's `(start_ns, end_ns)` window and project to the
    `nvml_state_at_launch` schema. Returns None if no in-window
    sample exists.

    The NVML sample carries `throttle_reasons` (the field name in
    `nvml_covariate.schema.json`); the dual-run-manifest schema's
    `nvml_state_at_launch` carries the same value as
    `throttle_reasons_bitmask`. The mapping is byte-for-byte; the
    rename is just naming-convention drift between the two schemas."""
    in_window = [
        s
        for s in nvml_samples_for_repeat
        if cupti_record.start_ns <= int(s["timestamp_ns"]) <= cupti_record.end_ns
    ]
    if not in_window:
        return None
    n = len(in_window)
    return {
        "sm_clock_mhz": int(
            round(sum(int(s["sm_clock_mhz"]) for s in in_window) / n)
        ),
        "mem_clock_mhz": int(
            round(sum(int(s["mem_clock_mhz"]) for s in in_window) / n)
        ),
        "temp_c": sum(float(s["temp_c"]) for s in in_window) / n,
        "power_mw": int(
            round(sum(int(s["power_mw"]) for s in in_window) / n)
        ),
        # Bitmask is the OR of the in-window samples (set-iff-any-set).
        "throttle_reasons_bitmask": int(
            max(int(s["throttle_reasons"]) for s in in_window)
        ),
        # Use the median sample's timestamp as the per-launch
        # representative; this is the value the canonical
        # nvml_state_divergence rule compares against.
        "sample_timestamp_ns_host_monotonic": int(
            in_window[len(in_window) // 2]["timestamp_ns"]
        ),
    }


# --- Manifest assembly ------------------------------------------------------


def _build_manifest(
    cupti: DualRunMetadata,
    nvbit: DualRunMetadata,
    *,
    drift_tolerance_ns: int,
    claim_scope: str = "measured_local",
) -> dict[str, Any]:
    matched_keys = sorted(
        set(cupti.launches_by_key.keys()) & set(nvbit.launches_by_key.keys())
    )
    launches: list[dict[str, Any]] = []
    for key in matched_keys:
        slug, shape_point, repeat_index, launch_index, kernel_signature_sha256 = key
        cupti_rec = cupti.launches_by_key[key]
        _ = nvbit.launches_by_key[key]  # confirm key matches NVBit side
        # The CUPTI launch's (start_ns, end_ns) IS the canonical
        # per-launch window in wall-clock terms; both runs' NVML
        # samples are aligned against the same window because the
        # two runs are back-to-back against bit-identical kernel
        # binaries (per the non-root measurement contract's dual-run protocol). The
        # canonical nvml_state_divergence rule then captures any
        # wall-clock drift between the two runs at the per-launch
        # boundary as the absolute delta between the two
        # `sample_timestamp_ns_host_monotonic` values.
        cupti_state = _aggregate_nvml_state_at_launch(
            cupti_rec, cupti.nvml_samples_by_repeat.get(repeat_index, [])
        )
        nvbit_state = _aggregate_nvml_state_at_launch(
            cupti_rec, nvbit.nvml_samples_by_repeat.get(repeat_index, [])
        )
        # Non-root measurement honesty rule: silent launch drops on missing
        # NVML alignment corrupt downstream pairing. Replace the
        # silent `continue` with a typed diagnostic naming the
        # offending launch + which side's NVML alignment failed.
        launch_match_key = {
            "slug": slug,
            "shape_point": shape_point,
            "repeat_index": repeat_index,
            "launch_index": launch_index,
            "kernel_signature_sha256": kernel_signature_sha256,
        }
        if cupti_state is None:
            raise DualRunNvmlAlignmentMissing(
                launch_match_key=launch_match_key, side="cupti"
            )
        if nvbit_state is None:
            raise DualRunNvmlAlignmentMissing(
                launch_match_key=launch_match_key, side="nvbit"
            )
        launches.append(
            {
                "launch_match_key": {
                    "slug": slug,
                    "shape_point": shape_point,
                    "repeat_index": repeat_index,
                    "launch_index": launch_index,
                    "kernel_signature_sha256": kernel_signature_sha256,
                },
                "cupti_run_nvml_state": cupti_state,
                "nvbit_run_nvml_state": nvbit_state,
            }
        )
    return {
        "schema_version": 1,
        "cupti_run_id": cupti.run_id,
        "nvbit_run_id": nvbit.run_id,
        "drift_tolerance_ns": drift_tolerance_ns,
        "binary_sha256": cupti.binary_sha256,
        "preflight_manifest_sha256": cupti.preflight_manifest_sha256,
        "launches": launches,
        "claim_scope": claim_scope,
    }


def _schema_check_manifest(manifest: dict[str, Any]) -> None:
    """Validate `manifest` against `dual_run_manifest.schema.json`
    using the existing schema-check route. Raises
    `DualRunManifestSchemaInvalid` on failure."""
    schema_path = _SCHEMAS_DIR / "dual_run_manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        _validate.validate(manifest, schema, root_path=schema_path)
    except _validate.ValidationError as e:
        raise DualRunManifestSchemaInvalid(
            schema_path=str(schema_path), validation_error=str(e)
        ) from e


# --- Public emitter ---------------------------------------------------------


def emit_dual_run_manifest(
    cupti_run_dir: Path,
    nvbit_run_dir: Path,
    output_path: Path,
    *,
    drift_tolerance_ns: int,
    claim_scope: str = "measured_local",
) -> dict[str, Any]:
    """Pair a CUPTI-side run directory with an NVBit-side run
    directory, build a schema-valid `dual_run_manifest.json`, and
    write it atomically. Returns the manifest dict for inspection.

    Pre-emission validation order (each step raises a typed
    diagnostic on defect; no output file is written until every
    check passes):

      1. Empty input dir -> DualRunManifestEmptyInputDir.
      2. binary_sha256 disagreement ->
         DualRunManifestBinarySha256Disagreement (re-raised from the
         consumer's assert_binary_sha256_consistency).
      3. preflight_manifest_sha256 disagreement ->
         DualRunPreflightDisagreement.
      4. unmatched per-launch keys ->
         DualRunManifestUnmatchedLaunches.
      5. NVBit-side missing NVML covariates ->
         DualRunNvbitNvmlMissing.
      6. Per-launch drift > drift_tolerance_ns ->
         DualRunDriftExceeded (re-raised from the consumer's
         admit_dual_run_manifest).
      7. Schema-check vs dual_run_manifest.schema.json ->
         DualRunManifestSchemaInvalid.

    The output is written via tmp+rename so a write failure does
    not leave a partial manifest on disk.
    """
    cupti = collect_run_metadata(cupti_run_dir, "cupti")
    nvbit = collect_run_metadata(nvbit_run_dir, "nvbit")

    _consumer.assert_binary_sha256_consistency(
        cupti.binary_sha256, nvbit.binary_sha256
    )
    # Check NVBit-NVML missing BEFORE preflight disagreement: when the
    # NVBit run has no NVML covariate sidecars, the preflight hash is
    # derived as `""` (since the phase ledger has no preflight field),
    # which would trigger a false-positive preflight disagreement.
    if nvbit.repeats_without_nvml:
        raise DualRunNvbitNvmlMissing(
            nvbit_run_dir=nvbit_run_dir,
            missing_repeat_indices=tuple(sorted(nvbit.repeats_without_nvml)),
        )
    if cupti.preflight_manifest_sha256 != nvbit.preflight_manifest_sha256:
        raise DualRunPreflightDisagreement(
            cupti_preflight_sha256=cupti.preflight_manifest_sha256,
            nvbit_preflight_sha256=nvbit.preflight_manifest_sha256,
        )

    cupti_keys = set(cupti.launches_by_key.keys())
    nvbit_keys = set(nvbit.launches_by_key.keys())
    unmatched_cupti = cupti_keys - nvbit_keys
    unmatched_nvbit = nvbit_keys - cupti_keys
    if unmatched_cupti or unmatched_nvbit:
        raise DualRunManifestUnmatchedLaunches(
            unmatched_cupti=tuple(
                {
                    "slug": k[0],
                    "shape_point": k[1],
                    "repeat_index": k[2],
                    "launch_index": k[3],
                    "kernel_signature_sha256": k[4],
                }
                for k in sorted(unmatched_cupti)
            ),
            unmatched_nvbit=tuple(
                {
                    "slug": k[0],
                    "shape_point": k[1],
                    "repeat_index": k[2],
                    "launch_index": k[3],
                    "kernel_signature_sha256": k[4],
                }
                for k in sorted(unmatched_nvbit)
            ),
        )

    manifest = _build_manifest(
        cupti,
        nvbit,
        drift_tolerance_ns=drift_tolerance_ns,
        claim_scope=claim_scope,
    )

    # Self-validate against the consumer (raises DualRunDriftExceeded
    # if any per-launch host-monotonic-timestamp delta exceeds the
    # tolerance per the canonical nvml_state_divergence rule).
    _consumer.admit_dual_run_manifest(manifest, drift_tolerance_ns=drift_tolerance_ns)

    # Schema-check before writing. The consumer's drift admission is
    # necessary but not sufficient: the schema layer pins the field
    # patterns + types + required-set that the consumer does not
    # re-check.
    _schema_check_manifest(manifest)

    # Atomic write: tmp + rename.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(output_path)
    return manifest
