"""Target-free construction of immutable ``sim_run_manifest.v2`` generations.

Production entry points are deliberately unavailable until external cohort,
existing Rust launch-admission, and build authorities are installed. The
separately named fixture lane exercises deterministic roster and publication
algebra without claiming that its projection created or authenticated launch
evidence. No code in this module opens labels, NCU reports, accuracy outputs,
runtime records, baseline databases, or calibration artifacts.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import InitVar, dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from jsonschema import Draft7Validator, Draft202012Validator, ValidationError
from referencing import Registry, Resource

from eval.contracts._common import AuthorityBlockerCode, authority_unavailable
from eval.contracts.launch_crosswalk import (
    CompleteLaunchKey,
    V2_FIXTURE_CONTRACT_MODE,
    WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID,
    WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256,
    verify_workload_execution_authority_fixture_model,
)
from eval.contracts.model_seal import RetainedPayloadSnapshot
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    ImmutableDirectoryGeneration,
    ImmutableFile,
    PinnedDirectoryReader,
    canonical_json_bytes,
    inspect_immutable_directory_generation,
    publish_directory,
    publish_regular_file,
    read_regular_snapshot,
    safe_relative_path,
    seal_model,
    sha256_bytes,
    strict_json_loads,
    verify_model_seal,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_SCHEMA_PATH = (
    ROOT / "eval/schemas/production_sim_manifest_authority.v1.schema.json"
)
RUN_MANIFEST_SCHEMA_PATH = ROOT / "sim/run/schema/sim_run_manifest.v2.schema.json"
LAUNCH_KEY_SCHEMA_PATH = ROOT / "eval/schemas/launch_key.v1.schema.json"
LAUNCH_EVIDENCE_SCHEMA_PATH = (
    ROOT / "sim/sm/schema/launch_evidence_manifest.v2.schema.json"
)
WORKLOAD_AUTHORITY_SCHEMA_PATH = (
    ROOT / "eval/schemas/workload_execution_authority.v1.schema.json"
)

AUTHORITY_SCHEMA_VERSION = 1
AUTHORITY_ARTIFACT_KIND = "EVALUATOR_PRODUCTION_SIM_MANIFEST_AUTHORITY"
AUTHORITY_SCHEMA_ID = (
    "https://gpu-simulator.local/eval/schemas/"
    "production_sim_manifest_authority.v1.schema.json"
)
AUTHORITY_SCHEMA_SHA256 = (
    "2c31f0640cb424df82be22b9b9692a5d9bae5196e4eb659828fd8e15c302197d"
)
RUN_MANIFEST_SCHEMA_ID = (
    "https://gpu-simulator.local/sim/run/schema/sim_run_manifest.v2.schema.json"
)
RUN_MANIFEST_SCHEMA_SHA256 = (
    "1e3e12c390806c5755402a21da249052b9bf75f3c990c84212260b9526fc00fb"
)
LAUNCH_KEY_SCHEMA_ID = (
    "https://gpu-simulator.local/eval/schemas/launch_key.v1.schema.json"
)
LAUNCH_KEY_SCHEMA_SHA256 = (
    "022b863ada5096841bfc0e1d83946c652d40c79a0fdb5e8910787a064648d15e"
)
LAUNCH_EVIDENCE_SCHEMA_ID = "launch_evidence_manifest.v2.schema.json"
LAUNCH_EVIDENCE_SCHEMA_SHA256 = (
    "1b21656e16f738c4d6363b96d7a8da864a383b52f6d7a6cfabaf8bfa1c8f0e03"
)
PRODUCER_ID = "eval.production_manifest"
PRODUCER_VERSION = 1
FIXTURE_CONTRACT_MODE = "FIXTURE_ONLY_NON_AUTHORITATIVE"
PRODUCTION_CONTRACT_MODE = "TARGET_FREE_PRODUCTION"
AUTHORITY_ROLE = "detached_target_free_sim_run_manifest_authority"
RUN_MANIFEST_NAME = "sim_run_manifest.v2.json"
FIXTURE_ADMISSION_KIND = "FIXTURE_EXISTING_RUST_LAUNCH_ADMISSION_PROJECTION"
FIXTURE_BUILD_KIND = "FIXTURE_DECLARED_SIMULATOR_BUILD_IDENTITY"
EXACT_COVERAGE = "EXACT_FULL_UNSAMPLED"
NO_BINARY_DIGEST_AUTHORITY = "NOT_CLAIMED_TRACE_CONTRACT_TYPED_GAP"
ARCH_ID = "sm_86"
SKU = "rtx_3090_ti"
FIXTURE_ARCH_CARD_PROJECTION_SHA256 = (
    "e1f38bedd65f78c8bc9bd7b79c49b21f1e0ac160c62efca1efb890591630a91c"
)
FIXTURE_PCHASE_CARD_PROJECTION_SHA256 = (
    "fb20b49c88f6e458e238d6a195955b0daae502a8c9b9cb3c9844faccd7086173"
)
ARCH_VERSION = 86
MAX_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_AUTHORITY_BYTES = 16 * 1024 * 1024

# These registries are independent trust inputs, not caller-populated options.
# They intentionally remain immutable and empty for current production evidence.
PRODUCTION_WORKLOAD_EXECUTION_AUTHORITIES = MappingProxyType({})
PRODUCTION_EXISTING_LAUNCH_ADMISSIONS = MappingProxyType({})
PRODUCTION_SIMULATOR_BUILD_AUTHORITIES = MappingProxyType({})
PRODUCTION_MANIFEST_PUBLICATION_AUTHORITIES = MappingProxyType({})

_VALIDATOR_SOURCE_PATHS = (
    "sim/sm/src/evidence.rs",
    "sim/trace/src/reader.rs",
    "sim/sm/src/control_bits.rs",
    "sim/sm/src/program.rs",
    "sim/arch/src/card.rs",
    "sim/mem/src/pchase_card.rs",
)
_SCHEMA_SPECS = (
    ("authority_schema", AUTHORITY_SCHEMA_PATH, AUTHORITY_SCHEMA_ID, AUTHORITY_SCHEMA_SHA256),
    (
        "sim_run_manifest_schema",
        RUN_MANIFEST_SCHEMA_PATH,
        RUN_MANIFEST_SCHEMA_ID,
        RUN_MANIFEST_SCHEMA_SHA256,
    ),
    ("launch_key_schema", LAUNCH_KEY_SCHEMA_PATH, LAUNCH_KEY_SCHEMA_ID, LAUNCH_KEY_SCHEMA_SHA256),
    (
        "launch_evidence_schema",
        LAUNCH_EVIDENCE_SCHEMA_PATH,
        LAUNCH_EVIDENCE_SCHEMA_ID,
        LAUNCH_EVIDENCE_SCHEMA_SHA256,
    ),
    (
        "workload_execution_authority_schema",
        WORKLOAD_AUTHORITY_SCHEMA_PATH,
        WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID,
        WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256,
    ),
)

_FORBIDDEN_JSON_KEYS = frozenset(
    {
        "accuracy",
        "accuracy_report",
        "baseline",
        "baseline_database",
        "baseline_runtime",
        "calibration",
        "label",
        "labels",
        "matched_label",
        "matched_labels",
        "metric_value",
        "metric_values",
        "ncu_report",
        "ncu_reports",
        "observed_value",
        "prediction",
        "predictions",
        "runtime_record",
        "same_cell_calibration",
        "score",
        "scores",
        "target",
        "target_value",
        "target_values",
        "unit_tagged_metrics",
    }
)
_FORBIDDEN_PATH_TOKEN_SEQUENCES = (
    ("accuracy",),
    ("baseline",),
    ("calibration",),
    ("label",),
    ("labels",),
    ("local", "label"),
    ("local", "labels"),
    ("matched", "label"),
    ("matched", "labels"),
    ("metric",),
    ("metrics",),
    ("ncu",),
    ("prediction",),
    ("predictions",),
    ("runtime",),
    ("score",),
    ("scores",),
    ("target",),
    ("target", "value"),
    ("target", "values"),
)
_FORBIDDEN_JSON_KEY_COMPACT = frozenset(
    key.replace("_", "") for key in _FORBIDDEN_JSON_KEYS
)
_FORBIDDEN_PATH_COMPACT = frozenset(
    "".join(tokens) for tokens in _FORBIDDEN_PATH_TOKEN_SEQUENCES
)
_INPUT_VALUE_KEYS = frozenset(
    {
        "argv",
        "binary",
        "directory",
        "directories",
        "file",
        "files",
        "filename",
        "path",
        "paths",
        "shape",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_PRIVATE_SOURCE_TOKEN = object()
_PRIVATE_WORKLOAD_TOKEN = object()
_PRIVATE_EVIDENCE_TOKEN = object()
_PRIVATE_BUILD_TOKEN = object()
_PRIVATE_MANIFEST_BUILD_TOKEN = object()
_PRIVATE_PUBLICATION_TOKEN = object()


class ProductionManifestError(ImmutableArtifactError):
    """A target-free manifest source, algebra, or publication failed."""

    def __init__(self, detail: str, *, code: str = "PRODUCTION_MANIFEST_INVALID") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


class SourceRole(str, Enum):
    WORKLOAD_AUTHORITY = "WORKLOAD_EXECUTION_AUTHORITY"
    BUILD_MANIFEST = "DECLARED_BUILD_MANIFEST"
    LAUNCH_EVIDENCE_MANIFEST = "LAUNCH_EVIDENCE_MANIFEST"
    DYNAMIC_TRACE = "DYNAMIC_NVBIT_TRACE"
    THREADBLOCK_PAYLOAD = "NVBIT_THREADBLOCK_PAYLOAD"
    ALLOCATIONS = "TRACE_ALLOCATIONS"
    STATS = "TRACE_STATS"
    EXECUTION_INFO = "STATIC_EXECUTION_INFO"
    CONTROL_BITS = "STATIC_CONTROL_BITS"
    ARCH_CARD = "SM86_ARCHITECTURE_CARD"
    PCHASE_CARD = "SM86_PCHASE_CARD"
    SCHEMA = "GOVERNING_SCHEMA"
    PRODUCER = "MANIFEST_PRODUCER"
    VALIDATOR = "EXISTING_RUST_VALIDATOR"


def _fail(
    detail: str, *, code: str = "PRODUCTION_MANIFEST_INVALID"
) -> None:
    raise ProductionManifestError(detail, code=code)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be lowercase 64-hex")
    return value


def _require_token(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) > 128
        or _TOKEN_RE.fullmatch(value) is None
    ):
        _fail(f"{label} must be a canonical ASCII token")
    _reject_forbidden_text(value, label)
    return value


def _normalized_tokens(value: str) -> tuple[str, ...]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", separated)
    return tuple(re.findall(r"[a-z0-9]+", separated.casefold()))


def _normalized_marker(value: str) -> str:
    return "-".join(_normalized_tokens(value))


def _reject_forbidden_text(
    value: str,
    label: str,
    *,
    allow_bare_ncu_contract_token: bool = False,
) -> None:
    tokens = _normalized_tokens(value)
    compact = "".join(tokens)
    for forbidden in _FORBIDDEN_PATH_TOKEN_SEQUENCES:
        if allow_bare_ncu_contract_token and forbidden == ("ncu",):
            continue
        width = len(forbidden)
        if any(
            tokens[index : index + width] == forbidden
            for index in range(len(tokens))
        ):
            _fail(
                f"{label} names a forbidden target, label, metric, score, "
                "baseline, runtime, or calibration surface"
            )
    compact_aliases = _FORBIDDEN_PATH_COMPACT | _FORBIDDEN_JSON_KEY_COMPACT
    if allow_bare_ncu_contract_token:
        compact_aliases = compact_aliases.difference({"ncu"})
    if compact in compact_aliases or any(
        token in compact_aliases for token in tokens
    ):
        _fail(f"{label} uses a forbidden compact target or label path alias")


def _input_value_key(key: str) -> bool:
    normalized = _normalized_marker(key).replace("-", "_")
    return normalized in _INPUT_VALUE_KEYS or normalized.endswith(
        ("_directory", "_directories", "_file", "_filename", "_path", "_paths")
    )


def _reject_absolute_artifact_values(value: object, label: str) -> None:
    if type(value) is str:
        if (
            re.search(r"(?:^|[\s='\"])(?:/|~/|\\\\|[A-Za-z]:[\\/])", value)
            is not None
            or re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", value) is not None
        ):
            _fail(f"{label} contains an absolute, home, drive, UNC, or URI value")
        if (
            re.search(r"(?:^|[\\/])\.\.(?:[\\/]|$)", value) is not None
            or value.strip() in {".", ".."}
        ):
            _fail(f"{label} contains path traversal")
        if "/" in value or "\\" in value:
            _reject_forbidden_text(value, label)
    elif type(value) is dict:
        for child in value.values():
            _reject_absolute_artifact_values(child, label)
    elif type(value) is list:
        for child in value:
            _reject_absolute_artifact_values(child, label)


def _reject_noncanonical_input_path(value: str, label: str) -> None:
    if "\\" in value or re.search(
        r"[A-Za-z][A-Za-z0-9+.-]*://",
        value,
    ) is not None:
        _fail(f"{label} contains an absolute, home, drive, UNC, or URI path")

    trimmed_end = len(value)
    while trimmed_end and value[trimmed_end - 1].isspace():
        trimmed_end -= 1

    candidate_pending = True
    for index, character in enumerate(value):
        if candidate_pending:
            if character.isspace():
                continue
            if character in {"/", "~", "\\"} or (
                index + 1 < len(value)
                and character.isascii()
                and character.isalpha()
                and value[index + 1] == ":"
            ):
                _fail(
                    f"{label} contains an absolute, home, drive, UNC, or URI path"
                )
            if character == ".":
                if (
                    index + 1 == trimmed_end
                    or (
                        index + 2 == trimmed_end
                        and value[index + 1] == "."
                    )
                    or value.startswith("../", index)
                ):
                    _fail(f"{label} contains path traversal")
                if value.startswith("./", index):
                    _fail(f"{label} contains a noncanonical relative path")
            candidate_pending = False
        if character in {"=", ":"}:
            candidate_pending = True

    candidate = value.strip()
    if not candidate or "/" not in candidate:
        return
    try:
        safe_relative_path(candidate)
    except ImmutableArtifactError as exc:
        raise ProductionManifestError(
            f"{label} contains a noncanonical relative path: {exc}"
        ) from exc


def _reject_input_value(value: object, label: str) -> None:
    if type(value) is str:
        _reject_forbidden_text(value, label)
        _reject_noncanonical_input_path(value, label)
    elif type(value) is list:
        for child in value:
            _reject_input_value(child, label)
    elif type(value) is dict:
        for child in value.values():
            _reject_input_value(child, label)


def _reject_target_surface(value: object, label: str) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{label} contains a non-string field")
            normalized_key = "_".join(_normalized_tokens(key))
            compact_key = "".join(_normalized_tokens(key))
            if (
                normalized_key in _FORBIDDEN_JSON_KEYS
                or compact_key in _FORBIDDEN_JSON_KEY_COMPACT
            ):
                _fail(f"{label} contains forbidden field {key!r}")
            if _input_value_key(key):
                _reject_input_value(child, f"{label} field {key!r}")
            _reject_target_surface(child, label)
    elif type(value) is list:
        for child in value:
            _reject_target_surface(child, label)


def _gate_target_free_path(path: str | Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in absolute.parts:
        if component in {os.path.sep, ""}:
            continue
        _reject_forbidden_text(component, label)
    return absolute


def _reject_retained_artifact_surface(value: object, label: str) -> None:
    _reject_target_surface(value, label)
    _reject_absolute_artifact_values(value, label)


def _reject_workload_scalar_metadata(
    value: object,
    path: tuple[str, ...] = (),
) -> None:
    if path and path[0] in {"argv", "shape"}:
        return
    if type(value) is dict:
        for key, child in value.items():
            _reject_workload_scalar_metadata(child, (*path, key))
        return
    if type(value) is list:
        for child in value:
            _reject_workload_scalar_metadata(child, path)
        return
    if type(value) is not str:
        return

    contractual_literals = {
        ("contract_mode",): {
            FIXTURE_CONTRACT_MODE,
            PRODUCTION_CONTRACT_MODE,
        },
        ("issuance_claim", "phase"): {"BEFORE_LABEL_COLLECTION"},
        ("issuance_claim", "input_policy"): {
            "TARGET_FREE_NO_LABEL_OR_PREDICTION_INPUTS"
        },
    }
    if value in contractual_literals.get(path, set()):
        return

    label = f"workload execution authority field {'.'.join(path)!r}"
    _reject_forbidden_text(
        value,
        label,
        allow_bare_ncu_contract_token=(path[:2] == ("source_contracts", "ncu")),
    )


def _validate_workload_target_free_model(value: dict[str, Any]) -> None:
    _reject_target_surface(value, "workload execution authority")
    _reject_workload_scalar_metadata(value)
    _reject_absolute_artifact_values(value, "workload execution authority")
    for field in ("argv", "shape"):
        entries = value.get(field)
        if type(entries) is not list or any(type(item) is not str for item in entries):
            _fail(f"workload execution authority {field} is not an exact string array")
        for item in entries:
            _reject_input_value(item, f"workload execution authority {field}")


def _gate_source_path(path: Path, role: SourceRole) -> Path:
    if not isinstance(role, SourceRole):
        _fail("source role is invalid")
    return _gate_target_free_path(path, f"{role.value} path")


def _strict_json_value(payload: bytes, label: str) -> Any:
    try:
        return strict_json_loads(payload, label=label)
    except (UnicodeError, ImmutableArtifactError) as exc:
        raise ProductionManifestError(f"invalid {label}: {exc}") from exc


def _strict_object(payload: bytes, label: str) -> dict[str, Any]:
    value = _strict_json_value(payload, label)
    if type(value) is not dict:
        _fail(f"{label} must be a JSON object")
    return value


def _schema_object(payload: bytes, label: str) -> dict[str, Any]:
    """Decode a schema without narrowing its declarative integer domains."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        _fail(f"{label} contains non-finite JSON constant {token}")

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite JSON number")
        return value

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_float=finite_float,
            parse_constant=constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionManifestError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        _fail(f"{label} must be a JSON object")
    return value


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        _fail(f"{label} fields are not exact")
    return value


def _file_identity(relative_path: str, payload: bytes) -> dict[str, Any]:
    safe_relative_path(relative_path)
    return {
        "relative_path": relative_path,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


# SOURCE_TYPES
@dataclass(frozen=True, slots=True)
class RetainedTargetFreeSource:
    """One no-follow, single-link retained input with no serialized pathname."""

    source_id: str
    role: SourceRole
    relative_path: str
    payload: bytes
    sha256: str
    size_bytes: int
    metadata: tuple[int, int, int, int, int, int, int]
    _source_path: Path = field(repr=False, compare=False)
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_SOURCE_TOKEN:
            _fail("retained source must come from the no-follow retention API")
        _require_token(self.source_id, "source_id")
        if not isinstance(self.role, SourceRole):
            _fail("retained source role is invalid")
        safe_relative_path(self.relative_path)
        _reject_forbidden_text(self.relative_path, "retained relative path")
        if type(self.payload) is not bytes:
            _fail("retained source payload is not exact bytes")
        _require_sha256(self.sha256, "retained source SHA-256")
        if self.sha256 != sha256_bytes(self.payload) or self.size_bytes != len(self.payload):
            _fail("retained source payload identity mismatch")
        if type(self.metadata) is not tuple or len(self.metadata) != 7:
            _fail("retained source metadata is invalid")
        if any(type(item) is not int or item < 0 for item in self.metadata):
            _fail("retained source metadata contains an invalid value")
        if not stat.S_ISREG(self.metadata[2]) or self.metadata[3] != 1:
            _fail("retained source must be a single-link regular file")
        if self.metadata[4] != self.size_bytes:
            _fail("retained source metadata size mismatch")
        if not isinstance(self._source_path, Path) or not self._source_path.is_absolute():
            _fail("retained source pathname is invalid")

    def identity_model(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "role": self.role.value,
            **_file_identity(self.relative_path, self.payload),
        }

    def recheck(self) -> None:
        path = _gate_source_path(self._source_path, self.role)
        try:
            current = read_regular_snapshot(path, max_bytes=self.size_bytes)
        except (OSError, ImmutableArtifactError) as exc:
            raise ProductionManifestError(
                f"retained source is unavailable or drifted: {self.source_id}: {exc}"
            ) from exc
        if current.metadata != self.metadata or current.payload != self.payload:
            _fail(f"retained source hash/path generation drift: {self.source_id}")


def retain_target_free_source_fixture(
    path: str | Path,
    *,
    source_id: str,
    role: SourceRole,
    relative_path: str,
    maximum_bytes: int = MAX_SOURCE_BYTES,
) -> RetainedTargetFreeSource:
    """Retain one explicit fixture input after the deny gate, without following links."""

    _require_token(source_id, "source_id")
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        _fail("maximum source bytes must be positive")
    safe_relative_path(relative_path)
    _reject_forbidden_text(relative_path, "retained relative path")
    source = _gate_source_path(Path(path), role)
    try:
        snapshot = read_regular_snapshot(source, max_bytes=maximum_bytes)
    except (OSError, ImmutableArtifactError) as exc:
        raise ProductionManifestError(f"could not retain {role.value}: {exc}") from exc
    return RetainedTargetFreeSource(
        source_id=source_id,
        role=role,
        relative_path=relative_path,
        payload=snapshot.payload,
        sha256=sha256_bytes(snapshot.payload),
        size_bytes=len(snapshot.payload),
        metadata=snapshot.metadata,
        _source_path=source,
        _construction_token=_PRIVATE_SOURCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class FixtureExistingLaunchAdmission:
    """Fixture-only projection of an admission already performed by Rust."""

    document: dict[str, Any]
    payload: bytes
    payload_sha256: str

    def __post_init__(self) -> None:
        if canonical_json_bytes(self.document) != self.payload:
            _fail("fixture existing-admission projection is not canonical JSON")
        _require_sha256(self.payload_sha256, "fixture admission payload SHA-256")
        if self.payload_sha256 != sha256_bytes(self.payload):
            _fail("fixture admission payload SHA-256 mismatch")

    def recheck(self) -> dict[str, Any]:
        model = _strict_object(
            self.payload, "retained fixture existing-admission projection"
        )
        if (
            canonical_json_bytes(model) != self.payload
            or sha256_bytes(self.payload) != self.payload_sha256
            or model != self.document
        ):
            _fail("fixture existing-admission projection mutated after retention")
        verify_model_seal(model)
        return model

    @property
    def replay_identity(self) -> dict[str, Any]:
        return self.document["replay_authenticated_launch"]

    @property
    def launch_evidence_identity_sha256(self) -> str:
        return self.document["launch_evidence_identity_sha256"]


def load_fixture_existing_launch_admission_projection(
    payload: bytes,
) -> FixtureExistingLaunchAdmission:
    """Load a non-authoritative wrapper; this function does not perform admission."""

    model = _strict_object(payload, "fixture existing launch-admission projection")
    if canonical_json_bytes(model) != payload:
        _fail("fixture existing-admission projection must use canonical JSON bytes")
    _exact_keys(
        model,
        {
            "schema_version",
            "artifact_kind",
            "authority_status",
            "existing_admission_id",
            "source_content_manifest_id",
            "source_content_manifest_sha256",
            "workload_execution_authority_model_seal_sha256",
            "launch_evidence_manifest_sha256",
            "launch_evidence_identity_sha256",
            "replay_authenticated_launch",
            "workload_execution_executable_sha256",
            "architecture",
            "control_bits_sha256",
            "arch_card_sha256",
            "pchase_card_sha256",
            "coverage",
            "binary_digest_authority",
            "model_seal_sha256",
        },
        "fixture existing-admission projection",
    )
    if model["schema_version"] != 1 or model["artifact_kind"] != FIXTURE_ADMISSION_KIND:
        _fail("fixture existing-admission projection version or kind is unsupported")
    if model["authority_status"] != FIXTURE_CONTRACT_MODE:
        _fail("fixture existing-admission projection attempted to claim authority")
    _require_token(model["existing_admission_id"], "existing admission ID")
    _require_token(model["source_content_manifest_id"], "source content manifest ID")
    for name in (
        "source_content_manifest_sha256",
        "workload_execution_authority_model_seal_sha256",
        "launch_evidence_manifest_sha256",
        "launch_evidence_identity_sha256",
        "workload_execution_executable_sha256",
        "control_bits_sha256",
        "arch_card_sha256",
        "pchase_card_sha256",
    ):
        _require_sha256(model[name], name)
    if model["coverage"] != EXACT_COVERAGE:
        _fail("partial or sampled launch evidence is forbidden")
    if model["binary_digest_authority"] != NO_BINARY_DIGEST_AUTHORITY:
        _fail("fixture projection may not invent binary-digest authority")
    _reject_target_surface(model, "fixture existing-admission projection")
    verify_model_seal(model)
    _validate_replay_identity(model["replay_authenticated_launch"])
    if model["architecture"] != {"arch_id": ARCH_ID, "sku": SKU}:
        _fail("fixture existing-admission architecture is not exact sm_86/RTX 3090 Ti")
    return FixtureExistingLaunchAdmission(model, payload, sha256_bytes(payload))


@dataclass(frozen=True, slots=True)
class FixtureWorkloadExecutionAuthority:
    document: dict[str, Any]
    source: RetainedTargetFreeSource
    schema_source: RetainedTargetFreeSource
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_WORKLOAD_TOKEN:
            _fail("fixture workload authority must come from its retained loader")
        if self.source.role is not SourceRole.WORKLOAD_AUTHORITY:
            _fail("fixture workload authority source role mismatch")
        if canonical_json_bytes(self.document) != self.source.payload:
            _fail("fixture workload authority bytes are not canonical")

    def recheck(self) -> dict[str, Any]:
        self.source.recheck()
        self.schema_source.recheck()
        model = _strict_object(
            self.source.payload, "retained fixture workload execution authority"
        )
        if canonical_json_bytes(model) != self.source.payload or model != self.document:
            _fail("fixture workload authority model mutated after retention")
        schema_payload, schema_snapshot = RetainedPayloadSnapshot.read_path(
            WORKLOAD_AUTHORITY_SCHEMA_PATH
        )
        try:
            verify_workload_execution_authority_fixture_model(
                model,
                schema_bytes=schema_payload,
                schema_snapshot=schema_snapshot,
            )
        except ImmutableArtifactError as exc:
            raise ProductionManifestError(
                f"fixture workload authority recheck failed: {exc}"
            ) from exc
        _validate_workload_target_free_model(model)
        return model


@dataclass(frozen=True, slots=True)
class FixtureDeclaredBuildIdentity:
    document: dict[str, Any]
    source: RetainedTargetFreeSource
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_BUILD_TOKEN:
            _fail("fixture build identity must come from its retained loader")
        if self.source.role is not SourceRole.BUILD_MANIFEST:
            _fail("fixture build source role mismatch")
        if canonical_json_bytes(self.document) != self.source.payload:
            _fail("fixture build manifest bytes are not canonical")

    def authority_model(self) -> dict[str, Any]:
        model = self.recheck()
        return _fixture_build_authority_model(
            model, build_manifest_sha256=self.source.sha256
        )

    def recheck(self) -> dict[str, Any]:
        self.source.recheck()
        model = _strict_object(
            self.source.payload, "retained fixture declared build identity"
        )
        if canonical_json_bytes(model) != self.source.payload or model != self.document:
            _fail("fixture declared build model mutated after retention")
        _fixture_build_authority_model(
            model, build_manifest_sha256=self.source.sha256
        )
        return model


def _schema_retained_source(
    path: Path,
    *,
    name: str,
    schema_id: str,
    expected_sha256: str,
) -> RetainedTargetFreeSource:
    retained = retain_target_free_source_fixture(
        path,
        source_id=f"schema.{name}",
        role=SourceRole.SCHEMA,
        relative_path=f"schemas/{path.name}",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    if retained.sha256 != expected_sha256:
        _fail(f"{name} schema hash drift")
    schema = _schema_object(retained.payload, f"{name} schema")
    if schema.get("$id") != schema_id:
        _fail(f"{name} schema identity drift")
    return retained


def load_fixture_workload_execution_authority(
    path: str | Path,
) -> FixtureWorkloadExecutionAuthority:
    source = retain_target_free_source_fixture(
        path,
        source_id="fixture.workload-execution-authority",
        role=SourceRole.WORKLOAD_AUTHORITY,
        relative_path="authority_inputs/workload_execution_authority.v1.json",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    schema = _schema_retained_source(
        WORKLOAD_AUTHORITY_SCHEMA_PATH,
        name="workload-authority",
        schema_id=WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID,
        expected_sha256=WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256,
    )
    model = _strict_object(source.payload, "fixture workload execution authority")
    if canonical_json_bytes(model) != source.payload:
        _fail("fixture workload execution authority must use canonical JSON bytes")
    _validate_workload_target_free_model(model)
    schema_payload, schema_snapshot = RetainedPayloadSnapshot.read_path(
        WORKLOAD_AUTHORITY_SCHEMA_PATH
    )
    try:
        verify_workload_execution_authority_fixture_model(
            model,
            schema_bytes=schema_payload,
            schema_snapshot=schema_snapshot,
        )
    except ImmutableArtifactError as exc:
        raise ProductionManifestError(
            f"fixture workload execution authority failed validation: {exc}"
        ) from exc
    return FixtureWorkloadExecutionAuthority(
        model, source, schema, _construction_token=_PRIVATE_WORKLOAD_TOKEN
    )


def _fixture_build_authority_model(
    model: dict[str, Any], *, build_manifest_sha256: str
) -> dict[str, Any]:
    _exact_keys(
        model,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "build_id",
            "source_tree_sha256",
            "source_commit",
            "traced_workload_executable_sha256",
            "simulator_executable_sha256",
            "arch_id",
            "sku",
            "model_seal_sha256",
        },
        "fixture declared build identity",
    )
    if model["schema_version"] != 1 or model["artifact_kind"] != FIXTURE_BUILD_KIND:
        _fail("fixture declared build identity version or kind is unsupported")
    if model["status"] != FIXTURE_CONTRACT_MODE:
        _fail("fixture declared build identity attempted to claim authority")
    _require_token(model["build_id"], "build_id")
    for name in (
        "source_tree_sha256",
        "traced_workload_executable_sha256",
        "simulator_executable_sha256",
    ):
        _require_sha256(model[name], name)
    if type(model["source_commit"]) is not str or _COMMIT_RE.fullmatch(
        model["source_commit"]
    ) is None:
        _fail("declared source commit must be explicit 40- or 64-hex")
    if model["arch_id"] != ARCH_ID or model["sku"] != SKU:
        _fail("declared build architecture is not exact sm_86/RTX 3090 Ti")
    _reject_target_surface(model, "fixture declared build identity")
    verify_model_seal(model)
    return {
        "status": FIXTURE_CONTRACT_MODE,
        "build_id": model["build_id"],
        "build_manifest_sha256": _require_sha256(
            build_manifest_sha256, "fixture build-manifest SHA-256"
        ),
        "source_tree_sha256": model["source_tree_sha256"],
        "source_commit": model["source_commit"],
        "traced_workload_executable_sha256": model[
            "traced_workload_executable_sha256"
        ],
        "simulator_executable_sha256": model["simulator_executable_sha256"],
        "arch_id": ARCH_ID,
        "sku": SKU,
    }


def load_fixture_declared_build_identity(
    path: str | Path,
) -> FixtureDeclaredBuildIdentity:
    source = retain_target_free_source_fixture(
        path,
        source_id="fixture.declared-build",
        role=SourceRole.BUILD_MANIFEST,
        relative_path="authority_inputs/declared_build_identity.v1.json",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    model = _strict_object(source.payload, "fixture declared build identity")
    if canonical_json_bytes(model) != source.payload:
        _fail("fixture declared build identity must use canonical JSON bytes")
    _fixture_build_authority_model(model, build_manifest_sha256=source.sha256)
    return FixtureDeclaredBuildIdentity(
        model, source, _construction_token=_PRIVATE_BUILD_TOKEN
    )


def _validate_replay_identity(value: object) -> dict[str, Any]:
    identity = _exact_keys(
        value,
        {
            "kernel_mangled_name",
            "grid",
            "block",
            "device_id",
            "stream_id",
            "binary_architecture",
        },
        "replay-authenticated launch identity",
    )
    if type(identity["kernel_mangled_name"]) is not str or not identity[
        "kernel_mangled_name"
    ]:
        _fail("replay-authenticated kernel name is empty")
    for name in ("grid", "block"):
        dim = _exact_keys(identity[name], {"x", "y", "z"}, name)
        if any(
            type(dim[axis]) is not int or not 1 <= dim[axis] <= (1 << 32) - 1
            for axis in ("x", "y", "z")
        ):
            _fail(f"replay-authenticated {name} is invalid")
    if type(identity["device_id"]) is not int or not 0 <= identity[
        "device_id"
    ] <= (1 << 32) - 1:
        _fail("replay-authenticated device_id is invalid")
    if type(identity["stream_id"]) is not int or not 0 <= identity[
        "stream_id"
    ] <= (1 << 53) - 1:
        _fail("replay-authenticated stream_id is invalid")
    if identity["binary_architecture"] != ARCH_ID:
        _fail("cross-architecture replay identity is forbidden")
    return identity


def _replay_projection(key: CompleteLaunchKey) -> dict[str, Any]:
    return {
        "kernel_mangled_name": key.kernel_mangled_name,
        "grid": {"x": key.grid[0], "y": key.grid[1], "z": key.grid[2]},
        "block": {"x": key.block[0], "y": key.block[1], "z": key.block[2]},
        "device_id": key.device_id,
        "stream_id": key.stream_id,
        "binary_architecture": key.binary_architecture,
    }


def _validate_replay_mode(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("replay configuration must be an exact object")
    mode = value.get("mode")
    if mode in {"flat", "l1_const"} and set(value) == {"mode"}:
        return copy.deepcopy(value)
    if (
        mode == "l1_shared_l2"
        and set(value) == {"mode", "dram_bandwidth"}
        and value["dram_bandwidth"] in {"carded", "unmodeled"}
    ):
        return copy.deepcopy(value)
    _fail("replay configuration is unsupported or not closed")


def _validate_json_schema(
    document: dict[str, Any], schema: dict[str, Any], *, label: str, draft7: bool = False
) -> None:
    validator_class = Draft7Validator if draft7 else Draft202012Validator
    try:
        validator_class(schema).validate(document)
    except ValidationError as exc:
        location = ".".join(str(item) for item in exc.absolute_path) or "$"
        _fail(f"{label} schema validation failed at {location}: {exc.message}")


def _positive_number(value: object, label: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not number > 0 or not number < float("inf"):
        _fail(f"{label} must be positive and finite")
    return number


def _validate_arch_card(payload: bytes) -> None:
    if sha256_bytes(payload) != FIXTURE_ARCH_CARD_PROJECTION_SHA256:
        _fail(
            "architecture card is not the exact clean fixture projection",
            code="CARD_PROJECTION_IDENTITY_MISMATCH",
        )
    card = _strict_object(payload, "sm_86 architecture card")
    _reject_retained_artifact_surface(card, "sm_86 architecture card")
    if (
        card.get("schema_version") != 1
        or card.get("artifact_kind")
        != "target_free_sim_arch_card_projection"
    ):
        _fail("architecture-card projection version or kind is unsupported")
    if card.get("arch_id") != ARCH_ID or card.get("sku") != SKU:
        _fail("architecture card is a cross-architecture or cross-SKU substitution")
    if card.get("compute_capability") != [8, 6]:
        _fail("architecture card compute capability is not exact sm_86")
    latencies = card.get("latencies_cycles")
    if type(latencies) is not dict:
        _fail("architecture card latency capability is unavailable")
    for name in ("l1_hit", "l2_hit", "dram", "shared"):
        entry = latencies.get(name)
        if type(entry) is not dict or "value" not in entry:
            _fail(f"architecture card required latency {name} is a research gap")
        _positive_number(entry["value"], f"architecture card {name} latency")


    l2 = card.get("l2_geometry")
    if type(l2) is not dict:
        _fail("architecture card L2 geometry is a research gap")
    for name in ("capacity_bytes", "sector_bytes", "num_slices", "line_bytes"):
        if type(l2.get(name)) is not int or l2[name] < 1:
            _fail(f"architecture card L2 field {name} is unresolved")


    bandwidth = card.get("dram_bandwidth")
    if type(bandwidth) is not dict or "peak_bytes_per_second" not in bandwidth:
        _fail("architecture card DRAM bandwidth is a research gap")
    _positive_number(
        bandwidth["peak_bytes_per_second"],
        "architecture card DRAM bandwidth",
    )


def _positive_exact_integer(value: object, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or type(value) not in {int, float}:
        _fail(f"{label} must be a positive exact integer")
    if type(value) is float and (not math.isfinite(value) or not value.is_integer()):
        _fail(f"{label} must be a positive exact integer")
    integer = int(value)
    if integer < 1 or integer > maximum:
        _fail(f"{label} is outside its supported positive integer range")
    return integer


def _validate_pchase_card(payload: bytes) -> None:
    if sha256_bytes(payload) != FIXTURE_PCHASE_CARD_PROJECTION_SHA256:
        _fail(
            "P-chase card is not the exact clean fixture projection",
            code="CARD_PROJECTION_IDENTITY_MISMATCH",
        )
    card = _strict_object(payload, "sm_86 P-chase card")
    _reject_retained_artifact_surface(card, "sm_86 P-chase card")
    if (
        card.get("schema_version") != 1
        or card.get("artifact_kind")
        != "target_free_sim_pchase_card_projection"
    ):
        _fail("P-chase card schema version or artifact kind is unsupported")
    if card.get("arch") != ARCH_ID:
        _fail("P-chase card is a cross-architecture substitution")
    if type(card.get("device")) is not str or card["device"] != (
        "NVIDIA GeForce RTX 3090 Ti"
    ):
        _fail("P-chase card device is not the exact RTX 3090 Ti SKU")
    if "sku" in card and card["sku"] != SKU:
        _fail("P-chase card SKU is a cross-SKU substitution")
    build = card.get("build")
    if type(build) is not dict or build.get("nvcc_arch") != ARCH_ID:
        _fail("P-chase card build architecture is not exact sm_86")
    for name in ("binary_sha256", "source_sha256"):
        _require_sha256(build.get(name), f"P-chase build {name}")

    properties = card.get("properties")
    if type(properties) is not dict:
        _fail("P-chase card properties are unavailable")
    capacity = properties.get("l1_capacity_bytes")
    if type(capacity) is not dict or capacity.get("kind") != "value" or (
        "value" not in capacity
    ):
        _fail("P-chase required property l1_capacity_bytes is a research gap")
    if capacity.get("unit") != "bytes":
        _fail("P-chase L1 capacity unit is not bytes")
    capacity_bytes = _positive_exact_integer(
        capacity["value"], "P-chase l1_capacity_bytes", maximum=(1 << 53) - 1
    )
    if capacity_bytes % 128 != 0:
        _fail("P-chase L1 capacity is not a positive line multiple")

    latency_values: dict[str, float] = {}
    for name in ("l1_hit_latency_cycles", "l2_hit_latency_cycles"):
        entry = properties.get(name)
        if type(entry) is not dict or entry.get("kind") != "value" or (
            "value" not in entry
        ):
            _fail(f"P-chase required property {name} is a research gap")
        if entry.get("unit") != "cycles_per_dependent_load":
            _fail(f"P-chase {name} unit is unsupported")
        latency_values[name] = _positive_number(entry["value"], f"P-chase {name}")
    if latency_values["l1_hit_latency_cycles"] >= latency_values[
        "l2_hit_latency_cycles"
    ]:
        _fail("P-chase hit/miss latency ordering is invalid")

    for name in (
        "mshr_outstanding_misses",
        "l1_replacement_policy",
        "l1_associativity_ways",
    ):
        entry = properties.get(name)
        if type(entry) is not dict or entry.get("kind") not in {"value", "typed"}:
            _fail(f"P-chase property {name} is unresolved")
        if entry["kind"] == "typed":
            if entry.get("typed_reason") != "geometry_unresolved":
                _fail(f"P-chase property {name} has an unsupported research gap")
            continue
        if "value" not in entry:
            _fail(f"P-chase property {name} kind=value lacks a value payload")
        if name == "mshr_outstanding_misses":
            _positive_exact_integer(
                entry["value"], name, maximum=(1 << 32) - 1
            )
        elif name == "l1_replacement_policy":
            if type(entry["value"]) is not str or entry["value"] != "lru":
                _fail("P-chase replacement policy is not the supported lru enum")
        else:
            _positive_exact_integer(entry["value"], name, maximum=(1 << 32) - 1)
            _fail("P-chase valued set associativity is not modeled")


def _validate_control_sidecar(payload: bytes) -> None:
    sidecar = _strict_object(payload, "static control-bit sidecar")
    _reject_retained_artifact_surface(sidecar, "static control-bit sidecar")
    if set(sidecar) != {"arch", "binary", "functions", "parser_contract", "summary"}:
        _fail("control-bit sidecar top-level projection is not closed")
    if sidecar.get("arch") != ARCH_ID:
        _fail("control-bit sidecar is a cross-architecture substitution")
    if type(sidecar.get("binary")) is not str or not sidecar["binary"]:
        _fail("control-bit sidecar binary identity is invalid")
    expected_layout = {
        "source": "high_word >> 41 (23 bits)",
        "stall": "bits 0:3",
        "yield_flag": "bit 4",
        "write_barrier": "bits 5:7 (7 = none)",
        "read_barrier": "bits 8:10 (7 = none)",
        "wait_mask": "bits 11:16",
        "reuse": "bits 17:20",
        "reserved_must_be_zero": "bits 21:22",
    }
    if sidecar.get("parser_contract") != {"control_layout": expected_layout}:
        _fail("control-bit sidecar parser contract is not exact")
    summary = sidecar.get("summary")
    if type(summary) is not dict or set(summary) != {
        "total_decoded",
        "total_unknown_encoding",
    }:
        _fail("control-bit sidecar summary projection is not closed")
    if any(type(value) is not int or value < 0 for value in summary.values()):
        _fail("control-bit sidecar summary counters are invalid")
    if summary["total_unknown_encoding"] != 0:
        _fail("control-bit sidecar carries unknown encodings")
    functions = sidecar.get("functions")
    if type(functions) is not dict or not functions:
        _fail("control-bit sidecar has no functions")
    decoded_rows = 0
    for function, rows in functions.items():
        if type(function) is not str or not function:
            _fail("control-bit sidecar function identity is invalid")
        if type(rows) is not list or not rows:
            _fail("control-bit sidecar contains an empty function schedule")
        observed_pcs: set[int] = set()
        for row in rows:
            if type(row) is not dict or set(row) != {
                "pc",
                "predicate",
                "opcode_text",
                "high_word",
                "status",
                "control",
            }:
                _fail("control-bit sidecar function row projection is not closed")
            if row.get("status") != "decoded":
                _fail("control-bit sidecar contains an unresolved control-bit gap")
            if type(row["pc"]) is not int or row["pc"] < 0 or row["pc"] in observed_pcs:
                _fail("control-bit sidecar function PCs are invalid or duplicated")
            observed_pcs.add(row["pc"])
            if row["predicate"] is not None and type(row["predicate"]) is not str:
                _fail("control-bit sidecar predicate is invalid")
            if type(row["opcode_text"]) is not str or not row["opcode_text"]:
                _fail("control-bit sidecar opcode text is invalid")
            if type(row["high_word"]) is not str or re.fullmatch(
                r"0x[0-9a-fA-F]{16}", row["high_word"]
            ) is None:
                _fail("control-bit sidecar high word is malformed")
            decoded_rows += 1
            control = row.get("control")
            expected_fields = {
                "stall",
                "yield_flag",
                "write_barrier",
                "read_barrier",
                "wait_mask",
                "reuse",
            }
            if type(control) is not dict or set(control) != expected_fields:
                _fail("control-bit sidecar decoded row has a non-exact control block")
            if any(type(control[name]) is not int for name in expected_fields):
                _fail("control-bit sidecar control fields are not exact integers")
            if not 0 <= control["stall"] <= 15 or control["yield_flag"] not in {0, 1}:
                _fail("control-bit sidecar stall/yield domain is invalid")
            for name in ("write_barrier", "read_barrier"):
                if control[name] not in {0, 1, 2, 3, 4, 5, 7}:
                    _fail("control-bit sidecar dependence-barrier domain is invalid")
            if not 0 <= control["wait_mask"] <= 0x3F or not 0 <= control[
                "reuse"
            ] <= 0x0F:
                _fail("control-bit sidecar wait/reuse domain is invalid")
    if summary["total_decoded"] != decoded_rows:
        _fail("control-bit sidecar summary count disagrees with decoded rows")


def _validate_trace_stats(payload: bytes) -> None:
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise ProductionManifestError("trace stats CSV is not exact ASCII") from exc
    _reject_forbidden_text(text, "trace stats CSV")
    try:
        rows = list(csv.reader(text.splitlines(), strict=True))
    except csv.Error as exc:
        raise ProductionManifestError(f"trace stats CSV is malformed: {exc}") from exc
    if not rows or rows[0] != ["total_insts", "total_reported_insts"]:
        _fail("trace stats CSV header is not the closed target-free schema")
    if len(rows) < 2:
        _fail("trace stats CSV has no data rows")
    expected_total = 0
    admitted_total = 0
    for row in rows[1:]:
        if len(row) != 2 or any(re.fullmatch(r"[0-9]+", cell) is None for cell in row):
            _fail("trace stats CSV row is not two exact unsigned integers")
        expected, admitted = (int(cell) for cell in row)
        if (
            expected > (1 << 64) - 1
            or admitted > (1 << 64) - 1
            or admitted > expected
        ):
            _fail("trace stats CSV counters are outside the admitted domain")
        expected_total += expected
        admitted_total += admitted
        if expected_total > (1 << 64) - 1 or admitted_total > (1 << 64) - 1:
            _fail("trace stats CSV aggregate counters overflow u64")
    if expected_total < 1 or admitted_total < 1:
        _fail("trace stats CSV counters are vacuous")


def _validate_allocations_sidecar(payload: bytes) -> None:
    value = _strict_json_value(payload, "trace allocations sidecar")
    _reject_retained_artifact_surface(value, "trace allocations sidecar")
    if type(value) is not list:
        _fail("trace allocations sidecar must be an exact array")
    expected = {
        "allocation_id",
        "context_id",
        "device_id",
        "base_address",
        "size_bytes",
        "alive",
    }
    observed_ids: set[int] = set()
    for row in value:
        if type(row) is not dict or set(row) != expected:
            _fail("trace allocation row fields are not exact")
        for name in ("allocation_id", "context_id", "device_id", "size_bytes"):
            if type(row[name]) is not int:
                _fail(f"trace allocation {name} is not an exact integer")
        if row["allocation_id"] < 1 or row["size_bytes"] < 1:
            _fail("trace allocation ID and size must be positive")
        if row["allocation_id"] in observed_ids:
            _fail("trace allocations sidecar repeats an allocation ID")
        observed_ids.add(row["allocation_id"])
        if type(row["alive"]) is not bool:
            _fail("trace allocation alive field is not boolean")
        address = row["base_address"]
        if type(address) is not str or re.fullmatch(r"0[xX][0-9a-fA-F]+", address) is None:
            _fail("trace allocation base address is not canonical hexadecimal")


def _validate_execution_info(
    payload: bytes, admission: FixtureExistingLaunchAdmission
) -> dict[str, Any]:
    info = _strict_object(payload, "enhanced execution information")
    _reject_retained_artifact_surface(info, "enhanced execution information")
    if set(info) != {"benchmark_name", "kernels"}:
        _fail("enhanced execution information top-level projection is not closed")
    if type(info["benchmark_name"]) is not str or not info["benchmark_name"]:
        _fail("enhanced execution information benchmark identity is invalid")
    kernels = info.get("kernels")
    if type(kernels) is not list or not kernels or any(
        type(row) is not dict for row in kernels
    ):
        _fail("enhanced execution information kernel roster is malformed")
    observed_ids: set[int] = set()
    for kernel in kernels:
        if set(kernel) != {
            "kernel_name",
            "unique_function_id",
            "architecture_version",
            "instructions",
        }:
            _fail("enhanced execution kernel projection is not closed")
        if type(kernel["kernel_name"]) is not str or not kernel["kernel_name"]:
            _fail("enhanced execution kernel name is invalid")
        function_id = kernel["unique_function_id"]
        if type(function_id) is not int or function_id < 0 or function_id in observed_ids:
            _fail("enhanced execution function identity is invalid or duplicated")
        observed_ids.add(function_id)
        if kernel["architecture_version"] != ARCH_VERSION:
            _fail("enhanced execution kernel architecture is not exact sm_86")
        instructions = kernel["instructions"]
        if type(instructions) is not list or not instructions:
            _fail("enhanced execution kernel instruction roster is empty")
        observed_pcs: set[int] = set()
        for instruction in instructions:
            if type(instruction) is not dict or set(instruction) != {
                "pc_string_hex",
                "pc_num_dec",
            }:
                _fail("enhanced execution instruction projection is not closed")
            pc = instruction["pc_num_dec"]
            text = instruction["pc_string_hex"]
            if (
                type(pc) is not int
                or pc < 0
                or pc in observed_pcs
                or type(text) is not str
                or re.fullmatch(r"[0-9a-f]+", text) is None
                or int(text, 16) != pc
            ):
                _fail("enhanced execution instruction PC is invalid or duplicated")
            observed_pcs.add(pc)
    identity = admission.replay_identity
    selected = [
        row
        for row in kernels
        if row.get("kernel_name") == identity["kernel_mangled_name"]
        and row.get("architecture_version") == ARCH_VERSION
    ]
    if len(selected) != 1:
        _fail("enhanced execution information launch identity is missing or duplicated")
    return info


@dataclass(frozen=True, slots=True)
class FixtureRetainedLaunchEvidence:
    """Retained files plus a fixture wrapper around existing Rust admission."""

    admission: FixtureExistingLaunchAdmission
    manifest_document: dict[str, Any]
    manifest_source: RetainedTargetFreeSource
    artifact_sources: tuple[RetainedTargetFreeSource, ...]
    retained_input_set_sha256: str
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_EVIDENCE_TOKEN:
            _fail("fixture launch evidence must come from its retained loader")
        if self.manifest_source.role is not SourceRole.LAUNCH_EVIDENCE_MANIFEST:
            _fail("launch-evidence manifest source role mismatch")
        _require_sha256(self.retained_input_set_sha256, "retained input-set SHA-256")

    @property
    def destination_manifest_path(self) -> str:
        return self.manifest_source.relative_path

    @property
    def all_sources(self) -> tuple[RetainedTargetFreeSource, ...]:
        return (self.manifest_source, *self.artifact_sources)

    def source_by_role(self, role: SourceRole) -> RetainedTargetFreeSource:
        selected = tuple(item for item in self.artifact_sources if item.role is role)
        if len(selected) != 1:
            _fail(f"launch evidence requires exactly one {role.value} source")
        return selected[0]

    def recheck(self) -> dict[str, Any]:
        admission = self.admission.recheck()
        for source in self.all_sources:
            source.recheck()
        manifest = _strict_object(
            self.manifest_source.payload, "retained launch-evidence manifest"
        )
        if manifest != self.manifest_document:
            _fail("retained launch-evidence model mutated after retention")
        if self.manifest_source.sha256 != admission[
            "launch_evidence_manifest_sha256"
        ]:
            _fail("retained admission/launch-evidence hash binding drifted")
        _reject_retained_artifact_surface(
            manifest, "retained launch-evidence manifest"
        )
        return manifest


def _artifact_claims(manifest: Mapping[str, Any]) -> dict[str, SourceRole]:
    artifacts = manifest["artifacts"]
    entries: list[tuple[str, Mapping[str, Any], SourceRole]] = [
        ("dynamic_trace_pb", artifacts["dynamic_trace_pb"], SourceRole.DYNAMIC_TRACE),
        ("stats_csv", artifacts["stats_csv"], SourceRole.STATS),
        (
            "enhanced_execution_info_json",
            artifacts["enhanced_execution_info_json"],
            SourceRole.EXECUTION_INFO,
        ),
        (
            "control_bits_sidecar",
            artifacts["control_bits_sidecar"],
            SourceRole.CONTROL_BITS,
        ),
        ("arch_card", artifacts["arch_card"], SourceRole.ARCH_CARD),
        ("pchase_card", artifacts["pchase_card"], SourceRole.PCHASE_CARD),
    ]
    optional = artifacts["allocations_json"]
    if optional["status"] == "present":
        entries.append(
            ("allocations_json", optional["artifact"], SourceRole.ALLOCATIONS)
        )
    if artifacts["trace_sampling_json"]["status"] != "absent":
        _fail("sampled or sampling-sidecar launch evidence is forbidden")

    paths: list[str] = []
    content_claims: list[tuple[int, str]] = []
    for name, claim, _role in entries:
        path = claim["path"]
        safe_relative_path(path)
        paths.append(path)
        content_claims.append((claim["size_bytes"], claim["sha256"]))
        if name == "dynamic_trace_pb" and not path.endswith("/dynamic_trace.pb"):
            _fail("dynamic trace role does not name its admitted protobuf artifact")
    if len(paths) != len(set(paths)):
        _fail("launch-evidence artifact roles duplicate a canonical path")
    if len(content_claims) != len(set(content_claims)):
        _fail("launch-evidence artifact roles duplicate a content claim")

    dynamic_path = safe_relative_path(artifacts["dynamic_trace_pb"]["path"])
    trace_root = dynamic_path.parent
    required_edges = {
        artifacts["threadblock_payload"]["path"]: trace_root.joinpath(
            "threadblocks"
        ).as_posix(),
        artifacts["stats_csv"]["path"]: trace_root.joinpath("stats.csv").as_posix(),
        artifacts["enhanced_execution_info_json"]["path"]: trace_root.joinpath(
            "extra_info/enhanced_execution_info.json"
        ).as_posix(),
    }
    if optional["status"] == "present":
        required_edges[optional["artifact"]["path"]] = trace_root.joinpath(
            "extra_info/allocations.json"
        ).as_posix()
    if any(actual != expected for actual, expected in required_edges.items()):
        _fail("launch-evidence trace artifact path edge is not canonical")
    threadblock_root = artifacts["threadblock_payload"]["path"]
    if any(path.startswith(f"{threadblock_root}/") for path in paths):
        _fail("ordinary launch artifact overlaps the threadblock payload root")
    return {claim["path"]: role for _name, claim, role in entries}


def _claim_for_path(manifest: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    for name in (
        "dynamic_trace_pb",
        "stats_csv",
        "enhanced_execution_info_json",
        "control_bits_sidecar",
        "arch_card",
        "pchase_card",
    ):
        claim = manifest["artifacts"][name]
        if claim["path"] == path:
            return claim
    optional = manifest["artifacts"]["allocations_json"]
    if optional["status"] == "present" and optional["artifact"]["path"] == path:
        return optional["artifact"]
    _fail(f"launch-evidence artifact path is not declared: {path}")


def _verify_threadblock_set(
    manifest: Mapping[str, Any], sources: Mapping[str, RetainedTargetFreeSource]
) -> None:
    claim = manifest["artifacts"]["threadblock_payload"]
    root = claim["path"]
    prefix = f"{root}/"
    rows = sorted(
        (path, source)
        for path, source in sources.items()
        if path.startswith(prefix)
    )
    if not rows or any(source.role is not SourceRole.THREADBLOCK_PAYLOAD for _, source in rows):
        _fail("threadblock payload set is empty or contains a wrong source role")
    relative_rows: list[tuple[str, RetainedTargetFreeSource]] = []
    trace_root = root.rsplit("/threadblocks", 1)[0]
    for path, source in rows:
        relative = path[len(trace_root) + 1 :]
        if not relative.startswith("threadblocks/") or not relative.endswith(".pb"):
            _fail("threadblock payload path is noncanonical")
        relative_rows.append((relative, source))
    manifest_text = "trace-payload-file-manifest-v1\n"
    stream = hashlib.sha256()
    raw_bytes = 0
    for relative, source in relative_rows:
        manifest_text += f"{relative}\t{source.size_bytes}\t{source.sha256}\n"
        stream.update(source.payload)
        raw_bytes += source.size_bytes
    if claim["file_count"] != len(relative_rows) or claim["raw_bytes"] != raw_bytes:
        _fail("threadblock payload set cardinality or byte count mismatch")
    if claim["sorted_path_manifest_sha256"] != sha256_bytes(manifest_text.encode()):
        _fail("threadblock sorted-path manifest hash mismatch")
    if claim["sorted_path_stream_sha256"] != stream.hexdigest():
        _fail("threadblock sorted-path stream hash mismatch")


def _retained_set_sha256(sources: Sequence[RetainedTargetFreeSource]) -> str:
    identities = [source.identity_model() for source in sources]
    identities.sort(key=lambda item: item["relative_path"])
    return sha256_bytes(
        canonical_json_bytes(
            {"domain": "target-free-retained-launch-input-set-v1", "files": identities}
        )
    )


def _launch_artifact_set_sha256(evidence: FixtureRetainedLaunchEvidence) -> str:
    artifacts = sorted(
        (source.role.value, source.sha256, source.size_bytes)
        for source in evidence.artifact_sources
    )
    return sha256_bytes(
        canonical_json_bytes(
            {"domain": "target-free-launch-artifact-set-v1", "artifacts": artifacts}
        )
    )


def _launch_inode_set_sha256(evidence: FixtureRetainedLaunchEvidence) -> str:
    inodes = sorted(
        (source.metadata[0], source.metadata[1]) for source in evidence.all_sources
    )
    return sha256_bytes(
        canonical_json_bytes(
            {"domain": "target-free-launch-source-inode-set-v1", "inodes": inodes}
        )
    )


def _launch_snapshot_sha256(evidence: FixtureRetainedLaunchEvidence) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "target-free-launch-source-snapshot-v2",
                "semantic_manifest_sha256": sha256_bytes(
                    canonical_json_bytes(evidence.manifest_document)
                ),
                "artifact_set_sha256": _launch_artifact_set_sha256(evidence),
            }
        )
    )


def retain_fixture_launch_evidence(
    evidence_manifest_path: str | Path,
    *,
    destination_manifest_path: str,
    artifact_paths: Mapping[str, str | Path],
    admission: FixtureExistingLaunchAdmission,
) -> FixtureRetainedLaunchEvidence:
    """Retain one full unsampled source set already projected by Rust admission."""

    if not isinstance(admission, FixtureExistingLaunchAdmission):
        _fail("fixture existing-admission projection is required")
    safe_relative_path(destination_manifest_path)
    _reject_forbidden_text(destination_manifest_path, "launch-evidence destination")
    manifest_source = retain_target_free_source_fixture(
        evidence_manifest_path,
        source_id=f"fixture.launch-evidence.{admission.document['existing_admission_id']}",
        role=SourceRole.LAUNCH_EVIDENCE_MANIFEST,
        relative_path=destination_manifest_path,
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _strict_object(manifest_source.payload, "launch-evidence manifest v2")
    _reject_retained_artifact_surface(manifest, "launch-evidence manifest v2")
    schema_source = _schema_retained_source(
        LAUNCH_EVIDENCE_SCHEMA_PATH,
        name="launch-evidence",
        schema_id=LAUNCH_EVIDENCE_SCHEMA_ID,
        expected_sha256=LAUNCH_EVIDENCE_SCHEMA_SHA256,
    )
    schema = _schema_object(schema_source.payload, "launch-evidence schema")
    _validate_json_schema(manifest, schema, label="launch-evidence manifest", draft7=True)
    schema_source.recheck()
    if manifest["schema_version"] != 2:
        _fail("unsupported launch-evidence manifest version")
    if manifest["architecture"] != {"arch_id": ARCH_ID, "sku": SKU}:
        _fail("launch-evidence manifest architecture is not exact sm_86/RTX 3090 Ti")
    if manifest["binary_digest"] != {
        "status": "typed_gap",
        "gap": "trace_contract_does_not_emit_binary_digest",
    }:
        _fail("launch-evidence binary digest must remain the typed trace-contract gap")
    if manifest_source.sha256 != admission.document["launch_evidence_manifest_sha256"]:
        _fail("existing admission does not bind the retained launch-evidence manifest")

    if not isinstance(artifact_paths, Mapping):
        _fail("artifact_paths must be an explicit manifest-relative mapping")
    raw_paths = dict(artifact_paths)
    if any(type(path) is not str for path in raw_paths):
        _fail("artifact_paths keys must be canonical strings")
    claims = _artifact_claims(manifest)
    threadblock_root = manifest["artifacts"]["threadblock_payload"]["path"]
    ordinary = set(claims)
    threadblocks = {path for path in raw_paths if path.startswith(f"{threadblock_root}/")}
    if set(raw_paths) != ordinary | threadblocks:
        _fail("artifact_paths is partial, contains extras, or omits a launch artifact")
    if not threadblocks:
        _fail("artifact_paths omits the complete threadblock payload set")

    destination_parent = Path(destination_manifest_path).parent.as_posix()
    destination_parent = "" if destination_parent == "." else destination_parent
    retained: list[RetainedTargetFreeSource] = []
    for index, relative in enumerate(sorted(raw_paths)):
        safe_relative_path(relative)
        role = claims.get(relative, SourceRole.THREADBLOCK_PAYLOAD)
        destination = f"{destination_parent}/{relative}" if destination_parent else relative
        retained.append(
            retain_target_free_source_fixture(
                raw_paths[relative],
                source_id=f"fixture.launch-artifact.{index}",
                role=role,
                relative_path=destination,
                maximum_bytes=MAX_SOURCE_BYTES,
            )
        )


    by_local_path = dict(zip(sorted(raw_paths), retained, strict=True))
    identities = [(item.metadata[0], item.metadata[1]) for item in (manifest_source, *retained)]
    if len(identities) != len(set(identities)):
        _fail("launch-evidence source files alias or use links")
    for relative, role in claims.items():
        source = by_local_path[relative]
        if source.role is not role:
            _fail(f"launch-evidence source role mismatch for {relative}")
        claim = _claim_for_path(manifest, relative)
        if claim["size_bytes"] != source.size_bytes or claim["sha256"] != source.sha256:
            _fail(f"launch-evidence hash/path drift for {relative}")
    _verify_threadblock_set(manifest, by_local_path)

    control = next(item for item in retained if item.role is SourceRole.CONTROL_BITS)
    arch_card = next(item for item in retained if item.role is SourceRole.ARCH_CARD)
    pchase = next(item for item in retained if item.role is SourceRole.PCHASE_CARD)
    execution = next(item for item in retained if item.role is SourceRole.EXECUTION_INFO)
    stats = next(item for item in retained if item.role is SourceRole.STATS)
    if control.sha256 != admission.document["control_bits_sha256"]:
        _fail("existing admission control-bit identity mismatch")
    if arch_card.sha256 != admission.document["arch_card_sha256"]:
        _fail("existing admission architecture-card identity mismatch")
    if pchase.sha256 != admission.document["pchase_card_sha256"]:
        _fail("existing admission P-chase-card identity mismatch")
    _validate_control_sidecar(control.payload)
    _validate_arch_card(arch_card.payload)
    _validate_pchase_card(pchase.payload)
    _validate_trace_stats(stats.payload)
    allocations = tuple(
        item for item in retained if item.role is SourceRole.ALLOCATIONS
    )
    if allocations:
        if len(allocations) != 1:
            _fail("launch evidence repeats the allocations sidecar")
        _validate_allocations_sidecar(allocations[0].payload)
    info = _validate_execution_info(execution.payload, admission)

    selector = manifest["launch_selector"]
    replay = admission.replay_identity
    if selector["device_id"] != replay["device_id"] or selector["stream_id"] != replay[
        "stream_id"
    ]:
        _fail("launch selector disagrees with replay-authenticated identity")
    selected = [
        row
        for row in info["kernels"]
        if row.get("kernel_name") == replay["kernel_mangled_name"]
        and row.get("architecture_version") == ARCH_VERSION
    ]
    if selected[0].get("unique_function_id") != selector["function_unique_id"]:
        _fail("launch selector function ID disagrees with execution metadata")

    material = (manifest_source, *retained)
    return FixtureRetainedLaunchEvidence(
        admission=admission,
        manifest_document=manifest,
        manifest_source=manifest_source,
        artifact_sources=tuple(retained),
        retained_input_set_sha256=_retained_set_sha256(material),
        _construction_token=_PRIVATE_EVIDENCE_TOKEN,
    )


def _load_governing_sources() -> tuple[
    dict[str, RetainedTargetFreeSource],
    RetainedTargetFreeSource,
    tuple[RetainedTargetFreeSource, ...],
]:
    schemas: dict[str, RetainedTargetFreeSource] = {}
    for name, path, schema_id, digest in _SCHEMA_SPECS:
        schemas[name] = _schema_retained_source(
            path,
            name=name,
            schema_id=schema_id,
            expected_sha256=digest,
        )
    producer = retain_target_free_source_fixture(
        Path(__file__),
        source_id="producer.eval-production-manifest",
        role=SourceRole.PRODUCER,
        relative_path="producer/eval.production_manifest.py",
        maximum_bytes=4 * 1024 * 1024,
    )
    validators = tuple(
        retain_target_free_source_fixture(
            ROOT / relative,
            source_id=f"validator.{index}",
            role=SourceRole.VALIDATOR,
            relative_path=relative,
            maximum_bytes=4 * 1024 * 1024,
        )
        for index, relative in enumerate(_VALIDATOR_SOURCE_PATHS)
    )
    return schemas, producer, validators


def _schema_identity_model(
    retained: RetainedTargetFreeSource, schema_id: str
) -> dict[str, Any]:
    return {
        "schema_id": schema_id,
        "sha256": retained.sha256,
        "size_bytes": retained.size_bytes,
    }


def _producer_identity_model(
    producer: RetainedTargetFreeSource,
    validators: Sequence[RetainedTargetFreeSource],
) -> dict[str, Any]:
    rows = [
        {
            "relative_path": item.relative_path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in validators
    ]
    bundle = sha256_bytes(
        canonical_json_bytes(
            {"domain": "existing-rust-manifest-validator-bundle-v1", "files": rows}
        )
    )
    return {
        "producer_id": PRODUCER_ID,
        "producer_version": PRODUCER_VERSION,
        "producer_sha256": producer.sha256,
        "producer_size_bytes": producer.size_bytes,
        "validator_bundle_sha256": bundle,
        "validator_sources": rows,
    }


@dataclass(frozen=True, slots=True)
class FixtureProductionManifestBuild:
    manifest_document: dict[str, Any]
    manifest_bytes: bytes
    generation_id: str
    generation_files: tuple[ImmutableFile, ...]
    source_identity: dict[str, Any]
    build_identity: dict[str, Any]
    producer_identity: dict[str, Any]
    schema_identities: dict[str, Any]
    authority_claims_bytes: bytes = field(repr=False)
    retained_sources: tuple[RetainedTargetFreeSource, ...] = field(repr=False)
    retained_wrappers: tuple[Any, ...] = field(repr=False)
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_MANIFEST_BUILD_TOKEN:
            _fail("fixture manifest build must come from the deterministic builder")
        _require_token(self.generation_id, "generation_id")
        if canonical_json_bytes(self.manifest_document) != self.manifest_bytes:
            _fail("simulator run manifest bytes are not canonical")
        if not self.generation_files or tuple(
            sorted(self.generation_files, key=lambda item: item.relative_path)
        ) != self.generation_files:
            _fail("generation file inventory is not sorted and complete")
        claims = _strict_object(self.authority_claims_bytes, "fixture build claims")
        if canonical_json_bytes(claims) != self.authority_claims_bytes or claims != {
            "source_identity": self.source_identity,
            "build_identity": self.build_identity,
            "producer_identity": self.producer_identity,
            "schema_identities": self.schema_identities,
        }:
            _fail("fixture build claim snapshot is inconsistent")

    def recheck(self) -> None:
        claims = _strict_object(self.authority_claims_bytes, "fixture build claims")
        if claims != {
            "source_identity": self.source_identity,
            "build_identity": self.build_identity,
            "producer_identity": self.producer_identity,
            "schema_identities": self.schema_identities,
        }:
            _fail("fixture build mutable claim fields changed after construction")
        if canonical_json_bytes(self.manifest_document) != self.manifest_bytes:
            _fail("fixture run-manifest model changed after construction")
        for wrapper in self.retained_wrappers:
            wrapper.recheck()
        for source in self.retained_sources:
            source.recheck()


def create_sim_run_manifest_fixture(
    *,
    generation_id: str,
    scope: str,
    workload_authority: FixtureWorkloadExecutionAuthority,
    launch_evidence: Sequence[FixtureRetainedLaunchEvidence],
    replay_mode: Mapping[str, Any],
    build_identity: FixtureDeclaredBuildIdentity,
) -> FixtureProductionManifestBuild:
    """Build exact v2 bytes from a complete explicit non-authoritative fixture set."""

    _require_token(generation_id, "generation_id")
    if type(scope) is not str or not scope or len(scope) > 256:
        _fail("manifest scope must be a nonempty bounded string")
    if not all(33 <= ord(character) <= 126 for character in scope):
        _fail("manifest scope must contain visible ASCII only")
    _reject_forbidden_text(scope, "manifest scope")
    if not isinstance(workload_authority, FixtureWorkloadExecutionAuthority):
        _fail("fixture workload execution authority is required")
    if not isinstance(build_identity, FixtureDeclaredBuildIdentity):
        _fail("fixture declared build identity is required")
    if type(launch_evidence) not in {list, tuple} or not launch_evidence:
        _fail("launch evidence must be a nonempty complete sequence")
    evidence = tuple(launch_evidence)
    if any(not isinstance(item, FixtureRetainedLaunchEvidence) for item in evidence):
        _fail("launch evidence sequence contains an invalid item")
    replay = _validate_replay_mode(replay_mode)
    workload = workload_authority.recheck()
    build_document = build_identity.recheck()
    for item in evidence:
        item.recheck()
    roster = workload["expected_launch_roster"]
    if workload["expected_launch_count"] != len(roster) or len(evidence) != len(roster):
        _fail("partial launch set disagrees with workload authority cardinality")
    keys = tuple(CompleteLaunchKey.from_model(row["launch_key"]) for row in roster)
    if tuple(key.process_launch_ordinal for key in keys) != tuple(range(len(keys))):
        _fail("launch roster is missing, duplicated, or reordered")
    key_models = [key.to_model() for key in keys]
    if len({canonical_json_bytes(item) for item in key_models}) != len(key_models):
        _fail("duplicate complete nine-field launch key")
    if workload["architecture"] != {
        "binary_architecture": ARCH_ID,
        "gpu_sku": SKU,
    }:
        _fail("workload authority architecture is not exact sm_86/RTX 3090 Ti")

    workload_executable = workload["executable_sha256"]
    if build_document["traced_workload_executable_sha256"] != workload_executable:
        _fail("declared build and workload executable hashes disagree")
    uniqueness_lanes = {
        "launch-evidence identity": [
            item.admission.launch_evidence_identity_sha256 for item in evidence
        ],
        "launch-evidence manifest hash": [item.manifest_source.sha256 for item in evidence],
        "semantic launch-evidence manifest": [
            sha256_bytes(canonical_json_bytes(item.manifest_document))
            for item in evidence
        ],
        "launch-evidence artifact set": [
            _launch_artifact_set_sha256(item) for item in evidence
        ],
        "launch-evidence retained inode set": [
            _launch_inode_set_sha256(item) for item in evidence
        ],
        "launch-evidence source snapshot": [
            _launch_snapshot_sha256(item) for item in evidence
        ],
    }
    for label, values in uniqueness_lanes.items():
        if len(values) != len(set(values)):
            _fail(f"{label} is reused across process launch ordinals")

    card_pairs: set[tuple[str, str]] = set()
    launch_rows: list[dict[str, Any]] = []
    manifest_launches: list[dict[str, Any]] = []
    destination_paths: set[str] = set()
    for index, (key, retained) in enumerate(zip(keys, evidence, strict=True)):
        admission = retained.admission.recheck()
        if admission["workload_execution_authority_model_seal_sha256"] != workload[
            "model_seal_sha256"
        ]:
            _fail("existing admission wrapper binds a different workload authority")
        if admission["workload_execution_executable_sha256"] != workload_executable:
            _fail("launch evidence and workload executable hashes disagree")
        if _replay_projection(key) != admission["replay_authenticated_launch"]:
            _fail("launch evidence disagrees with the complete nine-field roster/order")
        if retained.destination_manifest_path in destination_paths:
            _fail("duplicate launch-evidence manifest destination")
        destination_paths.add(retained.destination_manifest_path)
        card_pairs.add((admission["arch_card_sha256"], admission["pchase_card_sha256"]))
        manifest_launches.append(
            {
                "launch_evidence_manifest_path": retained.destination_manifest_path,
                "expected_identity_sha256": admission["launch_evidence_identity_sha256"],
                "launch_key": key.to_model(),
            }
        )
        launch_rows.append(
            {
                "launch_key": key.to_model(),
                "launch_key_sha256": key.sha256,
                "launch_evidence_manifest_path": retained.destination_manifest_path,
                "launch_evidence_manifest_sha256": retained.manifest_source.sha256,
                "launch_evidence_identity_sha256": admission[
                    "launch_evidence_identity_sha256"
                ],
                "existing_admission_id": admission["existing_admission_id"],
                "existing_admission_sha256": retained.admission.payload_sha256,
                "source_content_manifest_id": admission["source_content_manifest_id"],
                "source_content_manifest_sha256": admission[
                    "source_content_manifest_sha256"
                ],
                "retained_input_set_sha256": retained.retained_input_set_sha256,
                "control_bits_sha256": admission["control_bits_sha256"],
                "arch_card_sha256": admission["arch_card_sha256"],
                "pchase_card_sha256": admission["pchase_card_sha256"],
                "coverage": EXACT_COVERAGE,
            }
        )
    if len(card_pairs) != 1:
        _fail("launch set substitutes architecture or P-chase cards")


    arch_sha, pchase_sha = next(iter(card_pairs))
    roster_sha256 = sha256_bytes(canonical_json_bytes(key_models))
    replay_sha256 = sha256_bytes(canonical_json_bytes(replay))
    manifest = {
        "schema_version": 2,
        "scope": scope,
        "architecture": {"arch_id": ARCH_ID, "sku": SKU},
        "configuration_evidence": {
            "arch_card_sha256": arch_sha,
            "pchase_card_sha256": pchase_sha,
        },
        "launches": manifest_launches,
        "replay_mode": replay,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        _fail("simulator run manifest exceeds the retained bound")

    schemas, producer, validators = _load_governing_sources()
    run_schema = _schema_object(
        schemas["sim_run_manifest_schema"].payload, "sim-run manifest schema"
    )
    launch_key_schema = _schema_object(
        schemas["launch_key_schema"].payload, "launch-key schema"
    )
    registry = Registry().with_resource(
        LAUNCH_KEY_SCHEMA_ID, Resource.from_contents(launch_key_schema)
    )
    try:
        Draft202012Validator(run_schema, registry=registry).validate(manifest)
    except ValidationError as exc:
        location = ".".join(str(item) for item in exc.absolute_path) or "$"
        _fail(f"constructed sim-run manifest failed schema at {location}: {exc.message}")
    _reject_target_surface(manifest, "constructed sim-run manifest")

    schema_ids = {
        name: _schema_identity_model(schemas[name], schema_id)
        for name, _path, schema_id, _digest in _SCHEMA_SPECS
    }
    producer_model = _producer_identity_model(producer, validators)
    source_identity = {
        "workload_execution_authority": {
            "contract_mode": V2_FIXTURE_CONTRACT_MODE,
            "external_admission_status": "FIXTURE_ONLY_NON_AUTHORITATIVE",
            "payload_sha256": workload_authority.source.sha256,
            "size_bytes": workload_authority.source.size_bytes,
            "model_seal_sha256": workload["model_seal_sha256"],
        },
        "traced_workload_executable_sha256": workload_executable,
        "architecture": {"arch_id": ARCH_ID, "sku": SKU},
        "launch_count": len(keys),
        "launch_roster_sha256": roster_sha256,
        "launches": launch_rows,
        "replay_configuration_sha256": replay_sha256,
    }
    build_model = build_identity.authority_model()

    payloads: dict[str, bytes] = {
        RUN_MANIFEST_NAME: manifest_bytes,
        workload_authority.source.relative_path: workload_authority.source.payload,
        build_identity.source.relative_path: build_identity.source.payload,
        "authority_inputs/replay_configuration.v1.json": canonical_json_bytes(replay),
        producer.relative_path: producer.payload,
    }
    for source in (*schemas.values(), *validators):
        if source.relative_path in payloads:
            _fail(f"generation source path collision: {source.relative_path}")
        payloads[source.relative_path] = source.payload
    for retained in evidence:
        admission_path = (
            "authority_inputs/existing_admissions/"
            f"launch-{retained.admission.replay_identity['device_id']}-"
            f"{retained.admission.replay_identity['stream_id']}-"
            f"{retained.admission.document['existing_admission_id']}.json"
        )
        safe_relative_path(admission_path)
        if admission_path in payloads:
            _fail("duplicate existing-admission projection destination")
        payloads[admission_path] = retained.admission.payload
        for source in retained.all_sources:
            if source.relative_path in payloads:
                _fail(f"generation evidence path collision: {source.relative_path}")
            payloads[source.relative_path] = source.payload


    generation_files = tuple(
        ImmutableFile.from_bytes(path, payload)
        for path, payload in sorted(payloads.items())
    )
    retained_sources = (
        workload_authority.source,
        build_identity.source,
        producer,
        *schemas.values(),
        *validators,
        *(source for item in evidence for source in item.all_sources),
    )
    claims_bytes = canonical_json_bytes(
        {
            "source_identity": source_identity,
            "build_identity": build_model,
            "producer_identity": producer_model,
            "schema_identities": schema_ids,
        }
    )
    build = FixtureProductionManifestBuild(
        manifest_document=manifest,
        manifest_bytes=manifest_bytes,
        generation_id=generation_id,
        generation_files=generation_files,
        source_identity=source_identity,
        build_identity=build_model,
        producer_identity=producer_model,
        schema_identities=schema_ids,
        authority_claims_bytes=claims_bytes,
        retained_sources=tuple(retained_sources),
        retained_wrappers=(workload_authority, build_identity, *evidence),
        _construction_token=_PRIVATE_MANIFEST_BUILD_TOKEN,
    )
    build.recheck()
    return build


def verify_sim_run_manifest_fixture(
    build: FixtureProductionManifestBuild,
) -> None:
    """Recompute deterministic fixture algebra without opening any new source lane."""

    if not isinstance(build, FixtureProductionManifestBuild):
        _fail("fixture manifest build is invalid")
    build.recheck()
    if build.manifest_document["schema_version"] != 2:
        _fail("fixture manifest version drift")
    launches = build.manifest_document["launches"]
    tokens = [canonical_json_bytes(row["launch_key"]) for row in launches]
    if len(tokens) != len(set(tokens)):
        _fail("fixture manifest repeats a complete launch key")
    if [row["launch_key"]["process_launch_ordinal"] for row in launches] != list(
        range(len(launches))
    ):
        _fail("fixture manifest launch order is incomplete")
    if sha256_bytes(canonical_json_bytes([row["launch_key"] for row in launches])) != (
        build.source_identity["launch_roster_sha256"]
    ):
        _fail("fixture manifest roster seal mismatch")
    if canonical_json_bytes(build.manifest_document) != build.manifest_bytes:
        _fail("fixture manifest bytes drift")


def _generation_identity_model(
    generation: ImmutableDirectoryGeneration,
) -> dict[str, Any]:
    return {
        "content_manifest_contract": "immutable-directory-generation-v1",
        "content_manifest_sha256": generation.content_manifest_sha256,
        "filesystem_generation_sha256": generation.filesystem_generation_sha256,
        "file_count": generation.file_count,
        "directory_count": generation.directory_count,
        "total_file_bytes": generation.total_file_bytes,
        "files": [
            {
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in generation.files
        ],
    }


def _authority_document(
    build: FixtureProductionManifestBuild,
    generation: ImmutableDirectoryGeneration,
) -> dict[str, Any]:
    claims = _strict_object(build.authority_claims_bytes, "fixture build claims")
    manifest_identity = {
        "relative_path": RUN_MANIFEST_NAME,
        "schema_version": 2,
        "sha256": sha256_bytes(build.manifest_bytes),
        "size_bytes": len(build.manifest_bytes),
        "scope": build.manifest_document["scope"],
        "launch_roster_sha256": claims["source_identity"]["launch_roster_sha256"],
        "replay_configuration_sha256": claims["source_identity"][
            "replay_configuration_sha256"
        ],
        "replay_mode": build.manifest_document["replay_mode"],
    }
    document = seal_model(
        {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "artifact_kind": AUTHORITY_ARTIFACT_KIND,
            "schema_sha256": AUTHORITY_SCHEMA_SHA256,
            "contract_mode": FIXTURE_CONTRACT_MODE,
            "authority_role": AUTHORITY_ROLE,
            "generation_id": build.generation_id,
            "producer_identity": copy.deepcopy(claims["producer_identity"]),
            "schema_identities": copy.deepcopy(claims["schema_identities"]),
            "source_identity": copy.deepcopy(claims["source_identity"]),
            "build_identity": copy.deepcopy(claims["build_identity"]),
            "manifest_identity": manifest_identity,
            "generation_identity": _generation_identity_model(generation),
        }
    )
    _reject_target_surface(document, "detached simulator-manifest authority")
    _validate_authority_document(document)
    return document


def _validate_authority_document(document: dict[str, Any]) -> None:
    schema_payload, schema_snapshot = RetainedPayloadSnapshot.read_path(
        AUTHORITY_SCHEMA_PATH
    )
    schema_snapshot.require_production_acquisition("simulator-manifest authority schema")
    if schema_snapshot.payload_sha256 != AUTHORITY_SCHEMA_SHA256:
        _fail("detached simulator-manifest authority schema hash drift")
    schema = _schema_object(schema_payload, "simulator-manifest authority schema")
    _validate_json_schema(document, schema, label="simulator-manifest authority")
    verify_model_seal(document)
    if (
        document["schema_sha256"] != AUTHORITY_SCHEMA_SHA256
        or document["schema_version"] != AUTHORITY_SCHEMA_VERSION
        or document["artifact_kind"] != AUTHORITY_ARTIFACT_KIND
        or document["authority_role"] != AUTHORITY_ROLE
    ):
        _fail("detached authority top-level identity is not exact")
    if document["contract_mode"] != FIXTURE_CONTRACT_MODE:
        _fail("fixture authority attempted to claim production admission")
    _require_token(document["generation_id"], "detached authority generation ID")

    producer = document["producer_identity"]
    if (
        producer["producer_id"] != PRODUCER_ID
        or producer["producer_version"] != PRODUCER_VERSION
    ):
        _fail("detached authority producer identity is not exact")
    validator_rows = producer["validator_sources"]
    if [row["relative_path"] for row in validator_rows] != list(
        _VALIDATOR_SOURCE_PATHS
    ):
        _fail("detached authority validator source roster is not exact")
    validator_bundle = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "existing-rust-manifest-validator-bundle-v1",
                "files": validator_rows,
            }
        )
    )
    if producer["validator_bundle_sha256"] != validator_bundle:
        _fail("detached authority validator bundle seal is inconsistent")

    schema_identities = document["schema_identities"]
    for name, _path, schema_id, digest in _SCHEMA_SPECS:
        identity = schema_identities[name]
        if identity["schema_id"] != schema_id or identity["sha256"] != digest:
            _fail(f"detached authority {name} schema identity is not pinned")

    source = document["source_identity"]
    workload = source["workload_execution_authority"]
    if workload["contract_mode"] != V2_FIXTURE_CONTRACT_MODE or workload[
        "external_admission_status"
    ] != "FIXTURE_ONLY_NON_AUTHORITATIVE":
        _fail("fixture authority launders workload admission")
    if source["architecture"] != {"arch_id": ARCH_ID, "sku": SKU}:
        _fail("detached authority source architecture is not exact")
    build = document["build_identity"]
    if build["status"] != FIXTURE_CONTRACT_MODE:
        _fail("fixture authority launders build admission")
    if build["arch_id"] != ARCH_ID or build["sku"] != SKU:
        _fail("detached authority build architecture is not exact")
    if (
        source["traced_workload_executable_sha256"]
        != build["traced_workload_executable_sha256"]
    ):
        _fail("detached authority workload/build executable claims disagree")

    launches = source["launches"]
    if source["launch_count"] != len(launches) or not launches:
        _fail("detached authority launch count is not exact")
    keys: list[CompleteLaunchKey] = []
    evidence_paths: set[str] = set()
    for row in launches:
        key = CompleteLaunchKey.from_model(row["launch_key"])
        keys.append(key)
        if row["launch_key_sha256"] != key.sha256:
            _fail("detached authority launch-key seal is inconsistent")
        if row["coverage"] != EXACT_COVERAGE:
            _fail("detached authority launch coverage is partial or sampled")
        path = row["launch_evidence_manifest_path"]
        safe_relative_path(path)
        _reject_forbidden_text(path, "detached launch-evidence path")
        if path in evidence_paths:
            _fail("detached authority repeats a launch-evidence path")
        evidence_paths.add(path)
        _require_token(row["existing_admission_id"], "existing admission ID")
    if [key.process_launch_ordinal for key in keys] != list(range(len(keys))):
        _fail("detached authority launch roster is partial or reordered")
    key_models = [key.to_model() for key in keys]
    if len({canonical_json_bytes(key) for key in key_models}) != len(key_models):
        _fail("detached authority repeats a complete launch key")
    roster_sha256 = sha256_bytes(canonical_json_bytes(key_models))
    if source["launch_roster_sha256"] != roster_sha256:
        _fail("detached authority source launch-roster seal is inconsistent")
    card_pairs = {
        (row["arch_card_sha256"], row["pchase_card_sha256"])
        for row in launches
    }
    if len(card_pairs) != 1:
        _fail("detached authority substitutes configuration cards across launches")

    manifest = document["manifest_identity"]
    if manifest["relative_path"] != RUN_MANIFEST_NAME or manifest["schema_version"] != 2:
        _fail("detached authority run-manifest identity is not exact")
    if manifest["launch_roster_sha256"] != roster_sha256:
        _fail("detached authority source/run-manifest rosters disagree")
    replay = _validate_replay_mode(manifest["replay_mode"])
    replay_sha256 = sha256_bytes(canonical_json_bytes(replay))
    if (
        manifest["replay_configuration_sha256"] != replay_sha256
        or source["replay_configuration_sha256"] != replay_sha256
    ):
        _fail("detached authority replay-configuration seals disagree")
    _reject_forbidden_text(manifest["scope"], "detached run-manifest scope")

    generation = document["generation_identity"]
    files = generation["files"]
    paths = [row["relative_path"] for row in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail("detached authority generation inventory is not exact and sorted")
    for path in paths:
        safe_relative_path(path)
        _reject_forbidden_text(path, "detached generation file path")
    if generation["file_count"] != len(files) or generation[
        "total_file_bytes"
    ] != sum(row["size_bytes"] for row in files):
        _fail("detached authority generation inventory totals are inconsistent")
    by_path = {row["relative_path"]: row for row in files}
    manifest_file = by_path.get(RUN_MANIFEST_NAME)
    if manifest_file is None or {
        "sha256": manifest_file["sha256"],
        "size_bytes": manifest_file["size_bytes"],
    } != {"sha256": manifest["sha256"], "size_bytes": manifest["size_bytes"]}:
        _fail("detached authority run-manifest inventory claim is inconsistent")


@dataclass(frozen=True, slots=True)
class PublishedFixtureProductionManifest:
    generation_path: Path
    authority_path: Path
    generation: ImmutableDirectoryGeneration
    authority_document: dict[str, Any]
    authority_bytes: bytes
    authority_sha256: str
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_PUBLICATION_TOKEN:
            _fail("published fixture identity must come from the publisher")
        if canonical_json_bytes(self.authority_document) != self.authority_bytes:
            _fail("published detached authority bytes are not canonical")
        if sha256_bytes(self.authority_bytes) != self.authority_sha256:
            _fail("published detached authority file identity mismatch")


def publish_sim_run_manifest_fixture(
    build: FixtureProductionManifestBuild,
    output_root: str | Path,
) -> PublishedFixtureProductionManifest:
    """Publish one no-replace immutable generation and its detached authority."""

    verify_sim_run_manifest_fixture(build)
    root = _gate_target_free_path(output_root, "fixture publication path")
    try:
        generation_path = publish_directory(
            root,
            build.generation_id,
            build.generation_files,
            before_commit=lambda _root, _descriptor: build.recheck(),
            accept_existing=False,
        )
        generation = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=True,
            max_files=1_000_016,
            max_directories=1_000_032,
            max_total_file_bytes=4 * 1024 * 1024 * 1024,
            max_total_relative_path_bytes=64 * 1024 * 1024,
        )
        authority_document = _authority_document(build, generation)
        verified_generation = _verify_detached_document_generation(
            authority_document, generation_path
        )
        if verified_generation != generation:
            _fail("pre-publication authority rederivation changed generation identity")
        authority_bytes = canonical_json_bytes(authority_document)
        if len(authority_bytes) > MAX_AUTHORITY_BYTES:
            _fail("detached simulator-manifest authority exceeds the retained bound")
        authority_name = f"{build.generation_id}.authority.json"

        def recheck_publication() -> None:
            build.recheck()
            current = inspect_immutable_directory_generation(
                generation_path,
                require_nonempty=True,
                max_files=1_000_016,
                max_directories=1_000_032,
                max_total_file_bytes=4 * 1024 * 1024 * 1024,
                max_total_relative_path_bytes=64 * 1024 * 1024,
            )
            if current != generation:
                _fail(
                    "immutable simulator-manifest generation drifted before "
                    "authority publication"
                )
            if _verify_detached_document_generation(
                authority_document, generation_path
            ) != generation:
                _fail("detached authority precommit rederivation drifted")

        authority_path = publish_regular_file(
            root,
            authority_name,
            authority_bytes,
            before_commit=recheck_publication,
            after_commit=recheck_publication,
        )
        assert isinstance(authority_path, Path)
        reloaded = load_sim_run_manifest_authority_fixture(
            authority_path, generation_path
        )
        if reloaded != authority_document:
            _fail("published detached authority reload changed exact claim bytes")
    except ProductionManifestError:
        raise
    except (OSError, ImmutableArtifactError) as exc:
        raise ProductionManifestError(f"simulator-manifest publication failed: {exc}") from exc
    return PublishedFixtureProductionManifest(
        generation_path=generation_path,
        authority_path=authority_path,
        generation=generation,
        authority_document=authority_document,
        authority_bytes=authority_bytes,
        authority_sha256=sha256_bytes(authority_bytes),
        _construction_token=_PRIVATE_PUBLICATION_TOKEN,
    )


def _require_generation_file(
    by_path: Mapping[str, ImmutableFile],
    expected_paths: set[str],
    relative_path: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None,
    label: str,
) -> ImmutableFile:
    safe_relative_path(relative_path)
    if relative_path in expected_paths:
        _fail(f"generation authority reuses file path {relative_path}")
    expected_paths.add(relative_path)
    item = by_path.get(relative_path)
    if item is None or item.sha256 != expected_sha256:
        _fail(f"{label} is missing or has a different retained hash")
    if expected_size_bytes is not None and item.size_bytes != expected_size_bytes:
        _fail(f"{label} has a different retained size")
    return item


def _verify_generation_static_claims(
    document: Mapping[str, Any],
    by_path: Mapping[str, ImmutableFile],
    expected_paths: set[str],
    reader: PinnedDirectoryReader,
) -> None:
    def verify(
        relative_path: str, sha256: str, size_bytes: int, label: str
    ) -> None:
        item = _require_generation_file(
            by_path,
            expected_paths,
            relative_path,
            expected_sha256=sha256,
            expected_size_bytes=size_bytes,
            label=label,
        )
        payload = reader.read(relative_path, max_bytes=max(item.size_bytes, 1))
        if len(payload) != item.size_bytes or sha256_bytes(payload) != item.sha256:
            _fail(f"{label} bytes changed after generation inspection")

    producer = document["producer_identity"]
    verify(
        "producer/eval.production_manifest.py",
        producer["producer_sha256"],
        producer["producer_size_bytes"],
        "retained simulator-manifest producer",
    )
    for row in producer["validator_sources"]:
        verify(
            row["relative_path"],
            row["sha256"],
            row["size_bytes"],
            f"retained validator {row['relative_path']}",
        )
    for name, path, _schema_id, _digest in _SCHEMA_SPECS:
        row = document["schema_identities"][name]
        verify(
            f"schemas/{path.name}",
            row["sha256"],
            row["size_bytes"],
            f"retained schema {name}",
        )


def _verify_generation_authority_inputs(
    document: Mapping[str, Any],
    by_path: Mapping[str, ImmutableFile],
    expected_paths: set[str],
    reader: PinnedDirectoryReader,
    launch_key_payload: bytes,
) -> dict[str, Any]:
    source = document["source_identity"]
    workload_claim = source["workload_execution_authority"]
    workload_path = "authority_inputs/workload_execution_authority.v1.json"
    _require_generation_file(
        by_path,
        expected_paths,
        workload_path,
        expected_sha256=workload_claim["payload_sha256"],
        expected_size_bytes=workload_claim["size_bytes"],
        label="retained workload execution authority",
    )
    workload_payload = reader.read(workload_path, max_bytes=MAX_MANIFEST_BYTES)
    workload = _strict_object(workload_payload, "retained workload execution authority")
    if canonical_json_bytes(workload) != workload_payload:
        _fail("retained workload execution authority is not canonical JSON")
    _validate_workload_target_free_model(workload)
    verify_model_seal(workload)

    workload_schema_payload = reader.read(
        "schemas/workload_execution_authority.v1.schema.json",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    workload_schema = _schema_object(
        workload_schema_payload, "retained workload execution authority schema"
    )
    launch_schema = _schema_object(launch_key_payload, "retained launch-key schema")
    registry = Registry().with_resource(
        LAUNCH_KEY_SCHEMA_ID, Resource.from_contents(launch_schema)
    )
    try:
        Draft202012Validator(workload_schema, registry=registry).validate(workload)
    except ValidationError as exc:
        _fail(f"retained workload execution authority failed schema: {exc.message}")
    ambient_schema_payload, ambient_schema_snapshot = RetainedPayloadSnapshot.read_path(
        WORKLOAD_AUTHORITY_SCHEMA_PATH
    )
    if ambient_schema_payload != workload_schema_payload:
        _fail("retained workload schema differs from the independently pinned schema")
    try:
        verify_workload_execution_authority_fixture_model(
            workload,
            schema_bytes=ambient_schema_payload,
            schema_snapshot=ambient_schema_snapshot,
        )
    except ImmutableArtifactError as exc:
        raise ProductionManifestError(
            f"retained fixture workload authority failed full verification: {exc}"
        ) from exc
    if workload["model_seal_sha256"] != workload_claim["model_seal_sha256"]:
        _fail("retained workload authority model seal claim disagrees")
    if workload["executable_sha256"] != source[
        "traced_workload_executable_sha256"
    ]:
        _fail("retained workload executable claim disagrees")
    if workload["architecture"] != {
        "binary_architecture": ARCH_ID,
        "gpu_sku": SKU,
    }:
        _fail("retained workload architecture is not exact sm_86/RTX 3090 Ti")
    workload_keys = [row["launch_key"] for row in workload["expected_launch_roster"]]
    source_keys = [row["launch_key"] for row in source["launches"]]
    if (
        workload["expected_launch_count"] != len(source_keys)
        or workload_keys != source_keys
    ):
        _fail("retained workload launch roster disagrees with detached authority")

    build_path = "authority_inputs/declared_build_identity.v1.json"
    build_claim = document["build_identity"]
    _require_generation_file(
        by_path,
        expected_paths,
        build_path,
        expected_sha256=build_claim["build_manifest_sha256"],
        expected_size_bytes=None,
        label="retained declared build identity",
    )
    build_payload = reader.read(build_path, max_bytes=MAX_MANIFEST_BYTES)
    build_document = _strict_object(build_payload, "retained declared build identity")
    if canonical_json_bytes(build_document) != build_payload:
        _fail("retained declared build identity is not canonical JSON")
    derived_build = _fixture_build_authority_model(
        build_document, build_manifest_sha256=sha256_bytes(build_payload)
    )
    if derived_build != build_claim:
        _fail("retained declared build fields disagree with detached authority")

    replay_path = "authority_inputs/replay_configuration.v1.json"
    replay_sha256 = source["replay_configuration_sha256"]
    _require_generation_file(
        by_path,
        expected_paths,
        replay_path,
        expected_sha256=replay_sha256,
        expected_size_bytes=None,
        label="retained replay configuration",
    )
    replay_payload = reader.read(replay_path, max_bytes=MAX_MANIFEST_BYTES)
    replay_document = _strict_object(replay_payload, "retained replay configuration")
    if canonical_json_bytes(replay_document) != replay_payload:
        _fail("retained replay configuration is not canonical JSON")
    replay = _validate_replay_mode(replay_document)
    if replay != document["manifest_identity"]["replay_mode"]:
        _fail("retained replay configuration disagrees with run-manifest authority")
    return workload


def _generation_relative(parent: str, child: str) -> str:
    safe_relative_path(child)
    combined = f"{parent}/{child}" if parent else child
    safe_relative_path(combined)
    return combined


def _verify_generation_launch_sources(
    document: Mapping[str, Any],
    manifest: Mapping[str, Any],
    by_path: Mapping[str, ImmutableFile],
    expected_paths: set[str],
    reader: PinnedDirectoryReader,
    launch_evidence_schema_payload: bytes,
) -> None:
    source_launches = document["source_identity"]["launches"]
    expected_manifest_launches = [
        {
            "launch_evidence_manifest_path": row["launch_evidence_manifest_path"],
            "expected_identity_sha256": row["launch_evidence_identity_sha256"],
            "launch_key": row["launch_key"],
        }
        for row in source_launches
    ]
    if manifest["launches"] != expected_manifest_launches:
        _fail("retained run manifest launch rows disagree with source authority")
    evidence_schema = _schema_object(
        launch_evidence_schema_payload, "retained launch-evidence schema"
    )
    uniqueness_lanes: dict[str, list[str]] = {
        "launch-evidence identity": [],
        "semantic launch-evidence manifest": [],
        "launch-evidence artifact set": [],
        "launch-evidence retained inode set": [],
        "launch selector/context/ordinal": [],
    }
    for row in source_launches:
        key = CompleteLaunchKey.from_model(row["launch_key"])
        evidence_path = row["launch_evidence_manifest_path"]
        evidence_file = _require_generation_file(
            by_path,
            expected_paths,
            evidence_path,
            expected_sha256=row["launch_evidence_manifest_sha256"],
            expected_size_bytes=None,
            label="retained launch-evidence manifest",
        )
        evidence_snapshot = reader.read_snapshot(
            evidence_path, max_bytes=MAX_MANIFEST_BYTES
        )
        evidence_payload = evidence_snapshot.payload
        launch_inodes = [
            (evidence_snapshot.metadata[0], evidence_snapshot.metadata[1])
        ]
        if (
            len(evidence_payload) != evidence_file.size_bytes
            or sha256_bytes(evidence_payload) != evidence_file.sha256
        ):
            _fail("retained launch-evidence manifest bytes changed")
        evidence = _strict_object(evidence_payload, "retained launch-evidence manifest")
        _reject_retained_artifact_surface(evidence, "retained launch-evidence manifest")
        _validate_json_schema(
            evidence,
            evidence_schema,
            label="retained launch-evidence manifest",
            draft7=True,
        )
        if evidence["schema_version"] != 2 or evidence["architecture"] != {
            "arch_id": ARCH_ID,
            "sku": SKU,
        }:
            _fail("retained launch-evidence version or architecture is unsupported")
        if evidence["binary_digest"] != {
            "status": "typed_gap",
            "gap": "trace_contract_does_not_emit_binary_digest",
        }:
            _fail("retained launch evidence invents binary-digest authority")
        uniqueness_lanes["launch-evidence identity"].append(
            row["launch_evidence_identity_sha256"]
        )
        uniqueness_lanes["semantic launch-evidence manifest"].append(
            sha256_bytes(canonical_json_bytes(evidence))
        )
        uniqueness_lanes["launch selector/context/ordinal"].append(
            sha256_bytes(
                canonical_json_bytes(
                    {
                        "launch_key": key.to_model(),
                        "selector": evidence["launch_selector"],
                        "architecture": evidence["architecture"],
                    }
                )
            )
        )
        admission_path = (
            "authority_inputs/existing_admissions/"
            f"launch-{key.device_id}-{key.stream_id}-"
            f"{row['existing_admission_id']}.json"
        )
        admission_file = _require_generation_file(
            by_path,
            expected_paths,
            admission_path,
            expected_sha256=row["existing_admission_sha256"],
            expected_size_bytes=None,
            label="retained existing-admission projection",
        )
        admission_payload = reader.read(admission_path, max_bytes=MAX_MANIFEST_BYTES)
        if (
            len(admission_payload) != admission_file.size_bytes
            or sha256_bytes(admission_payload) != admission_file.sha256
        ):
            _fail("retained existing-admission bytes changed")
        admission_projection = load_fixture_existing_launch_admission_projection(
            admission_payload
        )
        admission = admission_projection.document
        if (
            admission["existing_admission_id"] != row["existing_admission_id"]
            or admission["launch_evidence_manifest_sha256"]
            != row["launch_evidence_manifest_sha256"]
            or admission["launch_evidence_identity_sha256"]
            != row["launch_evidence_identity_sha256"]
            or admission["replay_authenticated_launch"] != _replay_projection(key)
        ):
            _fail("retained admission projection disagrees with launch authority")
        for field in (
            "source_content_manifest_id",
            "source_content_manifest_sha256",
            "control_bits_sha256",
            "arch_card_sha256",
            "pchase_card_sha256",
            "coverage",
        ):
            if admission[field] != row[field]:
                _fail(f"retained admission projection disagrees on {field}")
        if (
            admission["workload_execution_authority_model_seal_sha256"]
            != document["source_identity"]["workload_execution_authority"][
                "model_seal_sha256"
            ]
            or admission["workload_execution_executable_sha256"]
            != document["source_identity"]["traced_workload_executable_sha256"]
        ):
            _fail("retained admission binds a different workload authority")
        claims = _artifact_claims(evidence)
        parent_path = Path(evidence_path).parent.as_posix()
        parent_path = "" if parent_path == "." else parent_path
        artifact_payloads: dict[SourceRole, bytes] = {}
        artifact_items: dict[str, ImmutableFile] = {}
        local_artifact_paths = set(claims)
        for local_path in sorted(claims):
            claim = _claim_for_path(evidence, local_path)
            destination = _generation_relative(parent_path, local_path)
            item = _require_generation_file(
                by_path,
                expected_paths,
                destination,
                expected_sha256=claim["sha256"],
                expected_size_bytes=claim["size_bytes"],
                label=f"retained launch artifact {local_path}",
            )
            role = claims[local_path]
            snapshot = reader.read_snapshot(
                destination, max_bytes=MAX_SOURCE_BYTES
            )
            payload = snapshot.payload
            launch_inodes.append((snapshot.metadata[0], snapshot.metadata[1]))
            if len(payload) != item.size_bytes or sha256_bytes(payload) != item.sha256:
                _fail(f"retained launch artifact bytes changed for {local_path}")
            artifact_payloads[role] = payload
            artifact_items[local_path] = item
        if sha256_bytes(artifact_payloads[SourceRole.CONTROL_BITS]) != row[
            "control_bits_sha256"
        ]:
            _fail("retained control-bit card hash disagrees with launch authority")
        if sha256_bytes(artifact_payloads[SourceRole.ARCH_CARD]) != row[
            "arch_card_sha256"
        ]:
            _fail("retained architecture-card hash disagrees with launch authority")
        if sha256_bytes(artifact_payloads[SourceRole.PCHASE_CARD]) != row[
            "pchase_card_sha256"
        ]:
            _fail("retained P-chase-card hash disagrees with launch authority")
        threadblock_claim = evidence["artifacts"]["threadblock_payload"]
        threadblock_root = threadblock_claim["path"]
        destination_root = _generation_relative(parent_path, threadblock_root)
        threadblock_destinations = sorted(
            path for path in by_path if path.startswith(f"{destination_root}/")
        )
        if not threadblock_destinations:
            _fail("retained generation omits the threadblock payload set")
        trace_root = threadblock_root.rsplit("/threadblocks", 1)[0]
        manifest_text = "trace-payload-file-manifest-v1\n"
        stream = hashlib.sha256()
        raw_bytes = 0
        for destination in threadblock_destinations:
            local_path = (
                destination[len(parent_path) + 1 :] if parent_path else destination
            )
            if not local_path.endswith(".pb"):
                _fail("retained threadblock payload path is noncanonical")
            item = _require_generation_file(
                by_path,
                expected_paths,
                destination,
                expected_sha256=by_path[destination].sha256,
                expected_size_bytes=by_path[destination].size_bytes,
                label=f"retained threadblock payload {local_path}",
            )
            snapshot = reader.read_snapshot(
                destination, max_bytes=MAX_SOURCE_BYTES
            )
            payload = snapshot.payload
            launch_inodes.append((snapshot.metadata[0], snapshot.metadata[1]))
            if len(payload) != item.size_bytes or sha256_bytes(payload) != item.sha256:
                _fail(f"retained threadblock payload bytes changed for {local_path}")
            relative = local_path[len(trace_root) + 1 :]
            manifest_text += f"{relative}\t{item.size_bytes}\t{item.sha256}\n"
            stream.update(payload)
            raw_bytes += item.size_bytes
            local_artifact_paths.add(local_path)
            artifact_items[local_path] = item
        if (
            threadblock_claim["file_count"] != len(threadblock_destinations)
            or threadblock_claim["raw_bytes"] != raw_bytes
            or threadblock_claim["sorted_path_manifest_sha256"]
            != sha256_bytes(manifest_text.encode())
            or threadblock_claim["sorted_path_stream_sha256"] != stream.hexdigest()
        ):
            _fail("retained threadblock payload-set identity disagrees")
        _validate_control_sidecar(artifact_payloads[SourceRole.CONTROL_BITS])
        _validate_arch_card(artifact_payloads[SourceRole.ARCH_CARD])
        _validate_pchase_card(artifact_payloads[SourceRole.PCHASE_CARD])
        _validate_trace_stats(artifact_payloads[SourceRole.STATS])
        if SourceRole.ALLOCATIONS in artifact_payloads:
            _validate_allocations_sidecar(artifact_payloads[SourceRole.ALLOCATIONS])
        info = _validate_execution_info(
            artifact_payloads[SourceRole.EXECUTION_INFO], admission_projection
        )
        selector = evidence["launch_selector"]
        replay = admission["replay_authenticated_launch"]
        selected = [
            item
            for item in info["kernels"]
            if item.get("kernel_name") == replay["kernel_mangled_name"]
            and item.get("architecture_version") == ARCH_VERSION
        ]
        if (
            selector["device_id"] != replay["device_id"]
            or selector["stream_id"] != replay["stream_id"]
            or selected[0].get("unique_function_id")
            != selector["function_unique_id"]
        ):
            _fail("retained launch selector disagrees with admitted launch identity")

        artifact_set = sorted(
            (
                claims.get(path, SourceRole.THREADBLOCK_PAYLOAD).value,
                item.sha256,
                item.size_bytes,
            )
            for path, item in artifact_items.items()
        )
        uniqueness_lanes["launch-evidence artifact set"].append(
            sha256_bytes(
                canonical_json_bytes(
                    {
                        "domain": "target-free-launch-artifact-set-v1",
                        "artifacts": artifact_set,
                    }
                )
            )
        )
        uniqueness_lanes["launch-evidence retained inode set"].append(
            sha256_bytes(
                canonical_json_bytes(
                    {
                        "domain": "target-free-launch-source-inode-set-v1",
                        "inodes": sorted(launch_inodes),
                    }
                )
            )
        )
        retained_identities = [
            {
                "source_id": f"fixture.launch-evidence.{row['existing_admission_id']}",
                "role": SourceRole.LAUNCH_EVIDENCE_MANIFEST.value,
                "relative_path": evidence_path,
                "sha256": evidence_file.sha256,
                "size_bytes": evidence_file.size_bytes,
            }
        ]
        for index, local_path in enumerate(sorted(local_artifact_paths)):
            item = artifact_items[local_path]
            role = claims.get(local_path, SourceRole.THREADBLOCK_PAYLOAD)
            retained_identities.append(
                {
                    "source_id": f"fixture.launch-artifact.{index}",
                    "role": role.value,
                    "relative_path": _generation_relative(parent_path, local_path),
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
            )
        retained_identities.sort(key=lambda item: item["relative_path"])
        retained_set_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "target-free-retained-launch-input-set-v1",
                    "files": retained_identities,
                }
            )
        )
        if retained_set_sha256 != row["retained_input_set_sha256"]:
            _fail("retained launch input-set seal disagrees with generation files")
    for label, values in uniqueness_lanes.items():
        if len(values) != len(set(values)):
            _fail(f"retained generation reuses {label} across launch ordinals")


def _verify_detached_document_generation(
    document: dict[str, Any], generation_source: Path
) -> ImmutableDirectoryGeneration:
    _validate_authority_document(document)
    generation = inspect_immutable_directory_generation(
        generation_source,
        require_nonempty=True,
        max_files=1_000_016,
        max_directories=1_000_032,
        max_total_file_bytes=4 * 1024 * 1024 * 1024,
        max_total_relative_path_bytes=64 * 1024 * 1024,
    )
    if _generation_identity_model(generation) != document["generation_identity"]:
        _fail("detached authority content-manifest/filesystem generation mismatch")
    if generation_source.name != document["generation_id"]:
        _fail("detached authority generation ID/path mismatch")
    by_path = {item.relative_path: item for item in generation.files}
    expected_paths: set[str] = set()
    manifest_identity = document["manifest_identity"]
    manifest_file = _require_generation_file(
        by_path,
        expected_paths,
        RUN_MANIFEST_NAME,
        expected_sha256=manifest_identity["sha256"],
        expected_size_bytes=manifest_identity["size_bytes"],
        label="retained sim-run manifest",
    )
    with PinnedDirectoryReader(generation_source) as reader:
        _verify_generation_static_claims(document, by_path, expected_paths, reader)
        manifest_snapshot = reader.read_snapshot(
            RUN_MANIFEST_NAME, max_bytes=MAX_MANIFEST_BYTES
        )
        manifest_payload = manifest_snapshot.payload
        if (
            len(manifest_payload) != manifest_file.size_bytes
            or sha256_bytes(manifest_payload) != manifest_file.sha256
        ):
            _fail("published sim-run manifest bytes disagree with retained identity")
        manifest = _strict_object(manifest_payload, "published sim-run manifest")
        if canonical_json_bytes(manifest) != manifest_payload:
            _fail("published sim-run manifest is not canonical JSON")
        schema_payload = reader.read(
            "schemas/sim_run_manifest.v2.schema.json", max_bytes=MAX_MANIFEST_BYTES
        )
        launch_key_payload = reader.read(
            "schemas/launch_key.v1.schema.json", max_bytes=MAX_MANIFEST_BYTES
        )
        launch_evidence_schema_payload = reader.read(
            "schemas/launch_evidence_manifest.v2.schema.json",
            max_bytes=MAX_MANIFEST_BYTES,
        )
        reader.verify_regular(RUN_MANIFEST_NAME, manifest_snapshot.metadata)
    if sha256_bytes(schema_payload) != RUN_MANIFEST_SCHEMA_SHA256:
        _fail("published run-manifest schema identity mismatch")
    if sha256_bytes(launch_key_payload) != LAUNCH_KEY_SCHEMA_SHA256:
        _fail("published launch-key schema identity mismatch")
    if sha256_bytes(launch_evidence_schema_payload) != LAUNCH_EVIDENCE_SCHEMA_SHA256:
        _fail("published launch-evidence schema identity mismatch")
    run_schema = _schema_object(schema_payload, "published run-manifest schema")
    launch_schema = _schema_object(launch_key_payload, "published launch-key schema")
    registry = Registry().with_resource(
        LAUNCH_KEY_SCHEMA_ID, Resource.from_contents(launch_schema)
    )
    try:
        Draft202012Validator(run_schema, registry=registry).validate(manifest)
    except ValidationError as exc:
        _fail(f"published run manifest schema validation failed: {exc.message}")
    _reject_retained_artifact_surface(manifest, "published sim-run manifest")
    if (
        manifest["scope"] != manifest_identity["scope"]
        or manifest["replay_mode"] != manifest_identity["replay_mode"]
        or manifest["architecture"] != {"arch_id": ARCH_ID, "sku": SKU}
    ):
        _fail("published run-manifest envelope disagrees with detached authority")
    roster = [item["launch_key"] for item in manifest["launches"]]
    if sha256_bytes(canonical_json_bytes(roster)) != manifest_identity[
        "launch_roster_sha256"
    ]:
        _fail("published run manifest roster seal mismatch")
    card_pairs = {
        (row["arch_card_sha256"], row["pchase_card_sha256"])
        for row in document["source_identity"]["launches"]
    }
    arch_sha256, pchase_sha256 = next(iter(card_pairs))
    if manifest["configuration_evidence"] != {
        "arch_card_sha256": arch_sha256,
        "pchase_card_sha256": pchase_sha256,
    }:
        _fail("published run-manifest card identities disagree with source authority")
    with PinnedDirectoryReader(generation_source) as reader:
        _verify_generation_authority_inputs(
            document, by_path, expected_paths, reader, launch_key_payload
        )
        _verify_generation_launch_sources(
            document,
            manifest,
            by_path,
            expected_paths,
            reader,
            launch_evidence_schema_payload,
        )
    if expected_paths != set(by_path):
        extras = sorted(set(by_path) - expected_paths)
        _fail(f"retained generation has unclaimed files: {extras[:3]}")
    closing = inspect_immutable_directory_generation(
        generation_source,
        require_nonempty=True,
        max_files=1_000_016,
        max_directories=1_000_032,
        max_total_file_bytes=4 * 1024 * 1024 * 1024,
        max_total_relative_path_bytes=64 * 1024 * 1024,
    )
    if closing != generation:
        _fail("retained generation changed after manifest parsing")
    return generation


def load_sim_run_manifest_authority_fixture(
    authority_path: str | Path,
    generation_path: str | Path,
) -> dict[str, Any]:
    """Authenticate a fixture authority against the exact immutable generation."""

    authority_source = _gate_target_free_path(
        authority_path, "fixture detached-authority path"
    )
    generation_source = _gate_target_free_path(
        generation_path, "fixture generation path"
    )
    try:
        retained = read_regular_snapshot(
            authority_source, max_bytes=MAX_AUTHORITY_BYTES
        )
    except (OSError, ImmutableArtifactError) as exc:
        raise ProductionManifestError(f"detached authority is unavailable: {exc}") from exc
    document = _strict_object(retained.payload, "detached simulator-manifest authority")
    if canonical_json_bytes(document) != retained.payload:
        _fail("detached simulator-manifest authority is not canonical JSON")
    _reject_target_surface(document, "detached simulator-manifest authority")
    _validate_authority_document(document)
    _verify_detached_document_generation(document, generation_source)
    closing_authority = read_regular_snapshot(
        authority_source, max_bytes=MAX_AUTHORITY_BYTES
    )
    if closing_authority != retained:
        _fail("detached authority changed during generation validation")
    return document


def _require_production_authorities() -> None:
    if not PRODUCTION_WORKLOAD_EXECUTION_AUTHORITIES:
        raise authority_unavailable(
            AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
            "no externally admitted current-M1 workload/execution authority is installed",
        )
    if not PRODUCTION_EXISTING_LAUNCH_ADMISSIONS:
        raise authority_unavailable(
            AuthorityBlockerCode.TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE,
            "no existing Rust launch-evidence admission authority is installed",
        )
    if not PRODUCTION_SIMULATOR_BUILD_AUTHORITIES:
        raise authority_unavailable(
            AuthorityBlockerCode.SIMULATOR_OUTPUT_AUTHORITY_UNAVAILABLE,
            "no externally authenticated simulator build authority is installed",
        )
    if not PRODUCTION_MANIFEST_PUBLICATION_AUTHORITIES:
        raise authority_unavailable(
            AuthorityBlockerCode.REVIEW_OPERATOR_TRUST_AUTHORITY_UNAVAILABLE,
            "no production manifest publication authority is installed",
        )


def create_production_sim_run_manifest(**_inputs: Any) -> dict[str, Any]:
    """Remain typed-unavailable before inspecting any caller source or pathname."""

    _require_production_authorities()
    raise ProductionManifestError("production simulator-manifest constructor is unavailable")


def publish_production_sim_run_manifest(**_inputs: Any) -> dict[str, Any]:
    """Remain typed-unavailable before opening output or scientific input paths."""

    _require_production_authorities()
    raise ProductionManifestError("production simulator-manifest publisher is unavailable")


def load_production_sim_run_manifest_authority(**_inputs: Any) -> dict[str, Any]:
    """Remain typed-unavailable before opening a detached authority or generation."""

    _require_production_authorities()
    raise ProductionManifestError("production simulator-manifest authority loader is unavailable")


# Stable descriptive aliases; none is exported through eval's import registry yet.
create_production_manifest = create_production_sim_run_manifest
publish_production_manifest = publish_production_sim_run_manifest
load_production_manifest_authority = load_production_sim_run_manifest_authority

