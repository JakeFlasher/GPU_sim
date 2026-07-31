"""Closed v1 prediction-seal receipt created before any label opening."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Sequence

from tools.local_labels.immutable_artifact import ImmutableArtifactError

from ._common import (
    AuthorityBlockerCode,
    authority_unavailable,
    evaluation_boundary_model,
    require_exact_evaluation_boundary,
)
from ._receipt_common import (
    IDENTITY_SCHEMA_VERSION,
    EvaluatorModelIdentity,
    ReceiptPhase,
    ReceiptValidationSchemas,
    RetainedSchemaInput,
    derive_domain_sha256,
    phase_chain_model,
    require_ascii_token,
    require_exact_keys,
    require_exact_phase_chain,
    require_safe_nonnegative_integer,
    require_sha256,
)
from .launch_crosswalk import (
    ARTIFACT_KIND as CROSSWALK_ARTIFACT_KIND,
    SCHEMA_ID as CROSSWALK_SCHEMA_ID,
    SCHEMA_SHA256 as CROSSWALK_SCHEMA_SHA256,
    SCHEMA_VERSION as CROSSWALK_SCHEMA_VERSION,
    LaunchNamespace,
    LaunchRecord,
    verify_launch_crosswalk_fixture_model,
)
from .model_seal import (
    require_unique_identities,
    seal_evaluator_model,
    strict_json_object,
    verify_evaluator_model,
    verify_retained_schema,
)
from .preregistration_receipt import (
    preregistration_fixture_chain_id,
    preregistration_fixture_identity,
    verify_preregistration_receipt_fixture_model,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "EVALUATOR_PREDICTION_SEAL_FIXTURE"
SCHEMA_ID = (
    "https://gpu-simulator.invalid/eval/schemas/"
    "prediction_seal_receipt.v1.schema.json"
)
SCHEMA_SHA256 = "89f7c500645513f4ca929e1bac9255a213cd98949f47ffd318453ea164a55b36"
FIXTURE_CONTRACT_MODE = "FIXTURE_ONLY"
FIXTURE_IDENTITY_AUTHORITY_STATUS = "FIXTURE_ONLY_NONAUTHORITATIVE"
PRODUCTION_COHORT_AUTHORITIES = MappingProxyType({})
PRODUCTION_SIMULATOR_OUTPUT_AUTHORITIES = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class SimulatorArtifactDigests:
    """Exact simulator build/config/input/output byte identities."""

    build_sha256: str
    config_sha256: str
    input_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.build_sha256, "build_sha256")
        require_sha256(self.config_sha256, "config_sha256")
        require_sha256(self.input_sha256, "input_sha256")
        require_sha256(self.output_sha256, "output_sha256")

    def to_model(self) -> dict[str, str]:
        return {
            "build_sha256": self.build_sha256,
            "config_sha256": self.config_sha256,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }

    @classmethod
    def from_model(cls, value: object) -> "SimulatorArtifactDigests":
        require_exact_keys(
            value,
            {"build_sha256", "config_sha256", "input_sha256", "output_sha256"},
            "simulator artifact digests",
        )
        assert isinstance(value, dict)
        return cls(
            build_sha256=value["build_sha256"],
            config_sha256=value["config_sha256"],
            input_sha256=value["input_sha256"],
            output_sha256=value["output_sha256"],
        )


@dataclass(frozen=True, slots=True)
class CohortCandidateIdentity:
    """A candidate identity only; it does not select or define a cohort."""

    candidate_id: str
    canonical_launch_identity_sha256: str
    candidate_record_sha256: str
    candidate_identity_sha256: str
    authority_status: str = FIXTURE_IDENTITY_AUTHORITY_STATUS
    identity_schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.identity_schema_version) is not int or self.identity_schema_version != 1:
            raise ImmutableArtifactError("candidate identity schema version is unsupported")
        if self.authority_status != FIXTURE_IDENTITY_AUTHORITY_STATUS:
            raise ImmutableArtifactError("candidate identity authority status is invalid")
        require_ascii_token(self.candidate_id, "candidate_id")
        require_sha256(
            self.canonical_launch_identity_sha256,
            "canonical_launch_identity_sha256",
        )
        require_sha256(self.candidate_record_sha256, "candidate_record_sha256")
        require_sha256(self.candidate_identity_sha256, "candidate_identity_sha256")
        if self.candidate_identity_sha256 != self._derived_sha256():
            raise ImmutableArtifactError("cohort-candidate identity SHA-256 mismatch")

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        canonical_launch_identity_sha256: str,
        candidate_record_sha256: str,
    ) -> "CohortCandidateIdentity":
        unsigned = {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "authority_status": FIXTURE_IDENTITY_AUTHORITY_STATUS,
            "candidate_id": candidate_id,
            "canonical_launch_identity_sha256": canonical_launch_identity_sha256,
            "candidate_record_sha256": candidate_record_sha256,
        }
        return cls(
            candidate_id=candidate_id,
            canonical_launch_identity_sha256=canonical_launch_identity_sha256,
            candidate_record_sha256=candidate_record_sha256,
            candidate_identity_sha256=derive_domain_sha256(
                "FIXTURE_COHORT_CANDIDATE_IDENTITY_V1", unsigned
            ),
        )

    def _derived_sha256(self) -> str:
        return derive_domain_sha256(
            "FIXTURE_COHORT_CANDIDATE_IDENTITY_V1",
            {
                "identity_schema_version": self.identity_schema_version,
                "authority_status": self.authority_status,
                "candidate_id": self.candidate_id,
                "canonical_launch_identity_sha256": self.canonical_launch_identity_sha256,
                "candidate_record_sha256": self.candidate_record_sha256,
            },
        )

    def to_model(self) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "authority_status": self.authority_status,
            "candidate_id": self.candidate_id,
            "canonical_launch_identity_sha256": self.canonical_launch_identity_sha256,
            "candidate_record_sha256": self.candidate_record_sha256,
            "candidate_identity_sha256": self.candidate_identity_sha256,
        }

    @classmethod
    def from_model(cls, value: object) -> "CohortCandidateIdentity":
        require_exact_keys(
            value,
            {
                "identity_schema_version",
                "authority_status",
                "candidate_id",
                "canonical_launch_identity_sha256",
                "candidate_record_sha256",
                "candidate_identity_sha256",
            },
            "cohort-candidate identity",
        )
        assert isinstance(value, dict)
        return cls(
            identity_schema_version=value["identity_schema_version"],
            authority_status=value["authority_status"],
            candidate_id=value["candidate_id"],
            canonical_launch_identity_sha256=value[
                "canonical_launch_identity_sha256"
            ],
            candidate_record_sha256=value["candidate_record_sha256"],
            candidate_identity_sha256=value["candidate_identity_sha256"],
        )


@dataclass(frozen=True, slots=True)
class ExpectedPredictionRowIdentity:
    """One expected output row identity, with no target or label value."""

    row_id: str
    cohort_candidate_identity_sha256: str
    canonical_launch_identity_sha256: str
    simulator_output_row_sha256: str
    expected_row_identity_sha256: str
    authority_status: str = FIXTURE_IDENTITY_AUTHORITY_STATUS
    identity_schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.identity_schema_version) is not int or self.identity_schema_version != 1:
            raise ImmutableArtifactError("expected-row identity schema version is unsupported")
        if self.authority_status != FIXTURE_IDENTITY_AUTHORITY_STATUS:
            raise ImmutableArtifactError("expected-row authority status is invalid")
        require_ascii_token(self.row_id, "row_id")
        require_sha256(
            self.cohort_candidate_identity_sha256,
            "cohort_candidate_identity_sha256",
        )
        require_sha256(
            self.canonical_launch_identity_sha256,
            "canonical_launch_identity_sha256",
        )
        require_sha256(
            self.simulator_output_row_sha256,
            "simulator_output_row_sha256",
        )
        require_sha256(
            self.expected_row_identity_sha256,
            "expected_row_identity_sha256",
        )
        if self.expected_row_identity_sha256 != self._derived_sha256():
            raise ImmutableArtifactError("expected prediction row identity SHA-256 mismatch")

    @classmethod
    def create(
        cls,
        *,
        row_id: str,
        cohort_candidate_identity_sha256: str,
        canonical_launch_identity_sha256: str,
        simulator_output_row_sha256: str,
    ) -> "ExpectedPredictionRowIdentity":
        unsigned = {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "authority_status": FIXTURE_IDENTITY_AUTHORITY_STATUS,
            "row_id": row_id,
            "cohort_candidate_identity_sha256": cohort_candidate_identity_sha256,
            "canonical_launch_identity_sha256": canonical_launch_identity_sha256,
            "simulator_output_row_sha256": simulator_output_row_sha256,
        }
        return cls(
            row_id=row_id,
            cohort_candidate_identity_sha256=cohort_candidate_identity_sha256,
            canonical_launch_identity_sha256=canonical_launch_identity_sha256,
            simulator_output_row_sha256=simulator_output_row_sha256,
            expected_row_identity_sha256=derive_domain_sha256(
                "FIXTURE_EXPECTED_PREDICTION_ROW_IDENTITY_V1", unsigned
            ),
        )

    def _derived_sha256(self) -> str:
        return derive_domain_sha256(
            "FIXTURE_EXPECTED_PREDICTION_ROW_IDENTITY_V1",
            {
                "identity_schema_version": self.identity_schema_version,
                "authority_status": self.authority_status,
                "row_id": self.row_id,
                "cohort_candidate_identity_sha256": self.cohort_candidate_identity_sha256,
                "canonical_launch_identity_sha256": self.canonical_launch_identity_sha256,
                "simulator_output_row_sha256": self.simulator_output_row_sha256,
            },
        )

    def to_model(self) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "authority_status": self.authority_status,
            "row_id": self.row_id,
            "cohort_candidate_identity_sha256": self.cohort_candidate_identity_sha256,
            "canonical_launch_identity_sha256": self.canonical_launch_identity_sha256,
            "simulator_output_row_sha256": self.simulator_output_row_sha256,
            "expected_row_identity_sha256": self.expected_row_identity_sha256,
        }

    @classmethod
    def from_model(cls, value: object) -> "ExpectedPredictionRowIdentity":
        require_exact_keys(
            value,
            {
                "identity_schema_version",
                "authority_status",
                "row_id",
                "cohort_candidate_identity_sha256",
                "canonical_launch_identity_sha256",
                "simulator_output_row_sha256",
                "expected_row_identity_sha256",
            },
            "expected prediction row identity",
        )
        assert isinstance(value, dict)
        return cls(
            identity_schema_version=value["identity_schema_version"],
            authority_status=value["authority_status"],
            row_id=value["row_id"],
            cohort_candidate_identity_sha256=value[
                "cohort_candidate_identity_sha256"
            ],
            canonical_launch_identity_sha256=value[
                "canonical_launch_identity_sha256"
            ],
            simulator_output_row_sha256=value["simulator_output_row_sha256"],
            expected_row_identity_sha256=value["expected_row_identity_sha256"],
        )


@dataclass(frozen=True, slots=True)
class FixtureSimulatorOutputManifest:
    """Canonical non-retained fixture manifest; never production output evidence."""

    entries: tuple[tuple[str, str], ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or not self.entries:
            raise ImmutableArtifactError("fixture simulator output manifest is empty")
        previous: str | None = None
        observed_rows: set[str] = set()
        observed_hashes: set[str] = set()
        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise ImmutableArtifactError("fixture output manifest entry is invalid")
            row_id, row_sha256 = entry
            require_ascii_token(row_id, "fixture simulator output row_id")
            require_sha256(row_sha256, "fixture simulator output row SHA-256")
            if previous is not None and row_id <= previous:
                raise ImmutableArtifactError(
                    "fixture simulator output manifest is not uniquely ordered"
                )
            if row_id in observed_rows or row_sha256 in observed_hashes:
                raise ImmutableArtifactError(
                    "fixture simulator output manifest contains duplicate identity"
                )
            previous = row_id
            observed_rows.add(row_id)
            observed_hashes.add(row_sha256)
        require_sha256(self.manifest_sha256, "fixture output manifest SHA-256")
        if self.manifest_sha256 != self._derived_sha256():
            raise ImmutableArtifactError("fixture output manifest SHA-256 mismatch")

    @classmethod
    def create(
        cls, rows: Iterable[ExpectedPredictionRowIdentity]
    ) -> "FixtureSimulatorOutputManifest":
        material = tuple(rows)
        if any(not isinstance(row, ExpectedPredictionRowIdentity) for row in material):
            raise ImmutableArtifactError("fixture output manifest row type is invalid")
        entries = tuple(
            sorted(
                (
                    (row.row_id, row.simulator_output_row_sha256)
                    for row in material
                ),
                key=lambda item: item[0],
            )
        )
        core = cls._core_model(entries)
        return cls(
            entries=entries,
            manifest_sha256=derive_domain_sha256(
                "FIXTURE_SIMULATOR_OUTPUT_MANIFEST_V1", core
            ),
        )

    @staticmethod
    def _core_model(entries: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        return {
            "manifest_schema_version": 1,
            "authority_status": "FIXTURE_ONLY_NONRETAINED",
            "entry_count": len(entries),
            "entries": [
                {"row_id": row_id, "row_sha256": row_sha256}
                for row_id, row_sha256 in entries
            ],
        }

    def _derived_sha256(self) -> str:
        return derive_domain_sha256(
            "FIXTURE_SIMULATOR_OUTPUT_MANIFEST_V1",
            self._core_model(self.entries),
        )

    def to_model(self) -> dict[str, Any]:
        return {
            **self._core_model(self.entries),
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_model(cls, value: object) -> "FixtureSimulatorOutputManifest":
        require_exact_keys(
            value,
            {
                "manifest_schema_version",
                "authority_status",
                "entry_count",
                "entries",
                "manifest_sha256",
            },
            "fixture simulator output manifest",
        )
        assert isinstance(value, dict)
        if (
            type(value["manifest_schema_version"]) is not int
            or value["manifest_schema_version"] != 1
        ):
            raise ImmutableArtifactError("fixture output manifest version is invalid")
        if value["authority_status"] != "FIXTURE_ONLY_NONRETAINED":
            raise ImmutableArtifactError("fixture output manifest status is invalid")
        raw_entries = value["entries"]
        if type(raw_entries) is not list:
            raise ImmutableArtifactError("fixture output manifest entries are invalid")
        entries: list[tuple[str, str]] = []
        for entry in raw_entries:
            require_exact_keys(entry, {"row_id", "row_sha256"}, "output manifest entry")
            assert isinstance(entry, dict)
            entries.append((entry["row_id"], entry["row_sha256"]))
        if type(value["entry_count"]) is not int or value["entry_count"] != len(entries):
            raise ImmutableArtifactError("fixture output manifest count mismatch")
        return cls(tuple(entries), value["manifest_sha256"])


def cycles_prediction_status_model() -> dict[str, str]:
    return {
        "descriptor_id": "exec_cycles_gpc_max",
        "metric_id": "gpc__cycles_elapsed.max",
        "implementation_status": "required_unimplemented",
        "evaluation_state": "pending",
        "outcome": "unavailable",
        "score_status": "unscored",
    }


def build_prediction_seal_receipt(**_inputs: Any) -> dict[str, Any]:
    raise authority_unavailable(
        AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
        "production candidate/cohort and simulator-output authorities are absent",
    )


def build_prediction_seal_receipt_fixture(
    *,
    preregistration_receipt: dict[str, Any],
    canonical_crosswalks: Iterable[dict[str, Any]],
    cohort_candidates: Iterable[CohortCandidateIdentity],
    simulator_artifacts: SimulatorArtifactDigests,
    simulator_output_manifest: FixtureSimulatorOutputManifest,
    expected_rows: Iterable[ExpectedPredictionRowIdentity],
    schemas: ReceiptValidationSchemas,
) -> dict[str, Any]:
    """Pure fixture builder; no labels or label readers are accepted."""

    _verify_schema_input(schemas)
    predecessor = preregistration_fixture_identity(
        preregistration_receipt, schemas=schemas
    )
    chain_id = preregistration_fixture_chain_id(
        preregistration_receipt, schemas=schemas
    )
    crosswalk_inventory = _crosswalk_inventory_model(
        canonical_crosswalks, schemas=schemas
    )
    candidate_inventory = fixture_candidate_inventory_model(
        cohort_candidates, crosswalk_inventory=crosswalk_inventory
    )
    preregistered_inventory = _parse_candidate_inventory(
        preregistration_receipt["fixture_candidate_inventory"],
        crosswalk_inventory=crosswalk_inventory,
    )
    if candidate_inventory != preregistered_inventory:
        raise ImmutableArtifactError(
            "prediction candidate inventory differs from preregistration"
        )
    row_inventory = _expected_row_inventory_model(
        expected_rows,
        candidate_inventory=candidate_inventory,
    )
    if not isinstance(simulator_artifacts, SimulatorArtifactDigests):
        raise ImmutableArtifactError("simulator artifact digest type is invalid")
    if not isinstance(simulator_output_manifest, FixtureSimulatorOutputManifest):
        raise ImmutableArtifactError("fixture simulator output manifest type is invalid")
    _require_manifest_row_coverage(
        simulator_output_manifest,
        row_inventory=row_inventory,
    )
    if simulator_artifacts.output_sha256 != simulator_output_manifest.manifest_sha256:
        raise ImmutableArtifactError(
            "simulator output digest does not equal canonical manifest digest"
        )

    return seal_evaluator_model(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "schema_sha256": SCHEMA_SHA256,
            "contract_mode": FIXTURE_CONTRACT_MODE,
            "production_eligibility": {
                "status": "UNAVAILABLE",
                "reason_codes": [
                    "COHORT_AUTHORITY_UNAVAILABLE",
                    "SIMULATOR_OUTPUT_AUTHORITY_UNAVAILABLE",
                ],
            },
            "evaluation_boundary": evaluation_boundary_model(),
            "phase_chain": phase_chain_model(
                phase=ReceiptPhase.PREDICTION_SEAL,
                chain_id_sha256=chain_id,
                predecessor=predecessor,
            ),
            "canonical_crosswalk_inventory": crosswalk_inventory,
            "cohort_candidate_inventory": candidate_inventory,
            "simulator_artifacts": simulator_artifacts.to_model(),
            "fixture_simulator_output_manifest": simulator_output_manifest.to_model(),
            "expected_row_inventory": row_inventory,
            "cycles_prediction": cycles_prediction_status_model(),
        }
    )


def verify_prediction_seal_receipt_model(
    model: dict[str, Any],
    **_inputs: Any,
) -> None:
    del model
    raise authority_unavailable(
        AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
        "production prediction verification lacks cohort and output authorities",
    )


def verify_prediction_seal_receipt_fixture_model(
    model: dict[str, Any],
    *,
    preregistration_receipt: dict[str, Any],
    canonical_crosswalks: Iterable[dict[str, Any]],
    schemas: ReceiptValidationSchemas,
) -> None:
    schema = _verify_schema_input(schemas)
    verify_evaluator_model(
        model,
        expected_schema_version=SCHEMA_VERSION,
        expected_artifact_kind=ARTIFACT_KIND,
        expected_schema_sha256=SCHEMA_SHA256,
        expected_schema_id=SCHEMA_ID,
        schema_bytes=schema.schema_bytes,
        schema_snapshot=schema.schema_snapshot,
    )
    require_exact_keys(
        model,
        {
            "schema_version",
            "artifact_kind",
            "schema_sha256",
            "contract_mode",
            "production_eligibility",
            "evaluation_boundary",
            "phase_chain",
            "canonical_crosswalk_inventory",
            "cohort_candidate_inventory",
            "simulator_artifacts",
            "fixture_simulator_output_manifest",
            "expected_row_inventory",
            "cycles_prediction",
            "model_seal_sha256",
        },
        "prediction-seal receipt",
    )
    if model["contract_mode"] != FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("prediction receipt is not fixture-only")
    if model["production_eligibility"] != {
        "status": "UNAVAILABLE",
        "reason_codes": [
            "COHORT_AUTHORITY_UNAVAILABLE",
            "SIMULATOR_OUTPUT_AUTHORITY_UNAVAILABLE",
        ],
    }:
        raise ImmutableArtifactError("prediction production blocker mismatch")
    require_exact_evaluation_boundary(model["evaluation_boundary"])
    predecessor = preregistration_fixture_identity(
        preregistration_receipt, schemas=schemas
    )
    chain_id = preregistration_fixture_chain_id(
        preregistration_receipt, schemas=schemas
    )
    require_exact_phase_chain(
        model["phase_chain"],
        phase=ReceiptPhase.PREDICTION_SEAL,
        chain_id_sha256=chain_id,
        predecessor=predecessor,
    )

    expected_crosswalk_inventory = _crosswalk_inventory_model(
        canonical_crosswalks, schemas=schemas
    )
    parsed_crosswalk_inventory = _parse_crosswalk_inventory(
        model["canonical_crosswalk_inventory"]
    )
    if parsed_crosswalk_inventory != expected_crosswalk_inventory:
        raise ImmutableArtifactError("canonical crosswalk inventory identity mismatch")
    candidate_inventory = _parse_candidate_inventory(
        model["cohort_candidate_inventory"],
        crosswalk_inventory=parsed_crosswalk_inventory,
    )
    preregistered_inventory = _parse_candidate_inventory(
        preregistration_receipt["fixture_candidate_inventory"],
        crosswalk_inventory=parsed_crosswalk_inventory,
    )
    if candidate_inventory != preregistered_inventory:
        raise ImmutableArtifactError(
            "prediction candidate inventory differs from preregistration"
        )
    row_inventory = _parse_expected_row_inventory(
        model["expected_row_inventory"],
        candidate_inventory=candidate_inventory,
    )
    simulator_artifacts = SimulatorArtifactDigests.from_model(
        model["simulator_artifacts"]
    )
    manifest = FixtureSimulatorOutputManifest.from_model(
        model["fixture_simulator_output_manifest"]
    )
    _require_manifest_row_coverage(manifest, row_inventory=row_inventory)
    if simulator_artifacts.output_sha256 != manifest.manifest_sha256:
        raise ImmutableArtifactError(
            "simulator output digest does not equal canonical manifest digest"
        )
    if model["cycles_prediction"] != cycles_prediction_status_model():
        raise ImmutableArtifactError("cycles prediction status is not pending/unavailable")


def load_prediction_seal_receipt(
    payload: bytes,
    **_inputs: Any,
) -> dict[str, Any]:
    del payload
    raise authority_unavailable(
        AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
        "production prediction loading lacks cohort and output authorities",
    )


def load_prediction_seal_receipt_fixture(
    payload: bytes,
    *,
    preregistration_receipt: dict[str, Any],
    canonical_crosswalks: Iterable[dict[str, Any]],
    schemas: ReceiptValidationSchemas,
) -> dict[str, Any]:
    model = strict_json_object(payload, label="evaluator prediction-seal fixture")
    verify_prediction_seal_receipt_fixture_model(
        model,
        preregistration_receipt=preregistration_receipt,
        canonical_crosswalks=canonical_crosswalks,
        schemas=schemas,
    )
    return model


def prediction_seal_fixture_identity(
    model: dict[str, Any],
    *,
    preregistration_receipt: dict[str, Any],
    canonical_crosswalks: Iterable[dict[str, Any]],
    schemas: ReceiptValidationSchemas,
) -> EvaluatorModelIdentity:
    verify_prediction_seal_receipt_fixture_model(
        model,
        preregistration_receipt=preregistration_receipt,
        canonical_crosswalks=canonical_crosswalks,
        schemas=schemas,
    )
    return EvaluatorModelIdentity.create(
        model,
        expected_schema_version=SCHEMA_VERSION,
        expected_artifact_kind=ARTIFACT_KIND,
        expected_schema_sha256=SCHEMA_SHA256,
        expected_schema_id=SCHEMA_ID,
        retained_schema=schemas.prediction_seal,
    )


def _crosswalk_inventory_model(
    canonical_crosswalks: Iterable[dict[str, Any]],
    *,
    schemas: ReceiptValidationSchemas,
) -> dict[str, Any]:
    if not isinstance(schemas, ReceiptValidationSchemas):
        raise ImmutableArtifactError("receipt validation schemas are invalid")
    entries: list[dict[str, Any]] = []
    for crosswalk in canonical_crosswalks:
        verify_launch_crosswalk_fixture_model(
            crosswalk,
            schema_bytes=schemas.launch_crosswalk.schema_bytes,
            schema_snapshot=schemas.launch_crosswalk.schema_snapshot,
            inventory_schema_bytes=schemas.launch_inventory.schema_bytes,
            inventory_schema_snapshot=schemas.launch_inventory.schema_snapshot,
        )
        model_identity = EvaluatorModelIdentity.create(
            crosswalk,
            expected_schema_version=CROSSWALK_SCHEMA_VERSION,
            expected_artifact_kind=CROSSWALK_ARTIFACT_KIND,
            expected_schema_sha256=CROSSWALK_SCHEMA_SHA256,
            expected_schema_id=CROSSWALK_SCHEMA_ID,
            retained_schema=schemas.launch_crosswalk,
        )
        entries.append(
            {
                "canonical_launch_identity_sha256": _canonical_launch_identity_sha256(
                    crosswalk
                ),
                "crosswalk_model": model_identity.to_model(),
            }
        )
    if not entries:
        raise ImmutableArtifactError("canonical crosswalk inventory must not be empty")
    material = require_unique_identities(
        entries,
        identity=lambda entry: entry["canonical_launch_identity_sha256"],
        label="canonical crosswalk launch",
    )
    require_unique_identities(
        material,
        identity=lambda entry: entry["crosswalk_model"]["canonical_model_sha256"],
        label="canonical crosswalk model",
    )
    ordered = sorted(
        material,
        key=lambda entry: (
            entry["canonical_launch_identity_sha256"],
            entry["crosswalk_model"]["canonical_model_sha256"],
        ),
    )
    core = {
        "inventory_schema_version": 1,
        "entry_count": len(ordered),
        "entries": ordered,
    }
    return {
        **core,
        "inventory_sha256": derive_domain_sha256(
            "FIXTURE_CANONICAL_CROSSWALK_INVENTORY_V1", core
        ),
    }


def _parse_crosswalk_inventory(value: object) -> dict[str, Any]:
    require_exact_keys(
        value,
        {"inventory_schema_version", "entry_count", "entries", "inventory_sha256"},
        "canonical crosswalk inventory",
    )
    assert isinstance(value, dict)
    if type(value["inventory_schema_version"]) is not int or value[
        "inventory_schema_version"
    ] != 1:
        raise ImmutableArtifactError("crosswalk inventory schema version is unsupported")
    entry_count = require_safe_nonnegative_integer(value["entry_count"], "entry_count")
    raw_entries = value["entries"]
    if type(raw_entries) is not list or not raw_entries:
        raise ImmutableArtifactError("canonical crosswalk entries must be a nonempty array")
    entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        require_exact_keys(
            raw_entry,
            {"canonical_launch_identity_sha256", "crosswalk_model"},
            "canonical crosswalk inventory entry",
        )
        assert isinstance(raw_entry, dict)
        launch_sha = require_sha256(
            raw_entry["canonical_launch_identity_sha256"],
            "canonical_launch_identity_sha256",
        )
        crosswalk_identity = EvaluatorModelIdentity.from_model(
            raw_entry["crosswalk_model"], label="canonical crosswalk model identity"
        )
        if (
            crosswalk_identity.schema_version != CROSSWALK_SCHEMA_VERSION
            or crosswalk_identity.artifact_kind != CROSSWALK_ARTIFACT_KIND
            or crosswalk_identity.schema_sha256 != CROSSWALK_SCHEMA_SHA256
        ):
            raise ImmutableArtifactError(
                "crosswalk model identity schema/version/kind mismatch"
            )
        entries.append(
            {
                "canonical_launch_identity_sha256": launch_sha,
                "crosswalk_model": crosswalk_identity.to_model(),
            }
        )
    if entry_count != len(entries):
        raise ImmutableArtifactError("crosswalk entry_count mismatch")
    require_unique_identities(
        entries,
        identity=lambda entry: entry["canonical_launch_identity_sha256"],
        label="canonical crosswalk launch",
    )
    require_unique_identities(
        entries,
        identity=lambda entry: entry["crosswalk_model"]["canonical_model_sha256"],
        label="canonical crosswalk model",
    )
    if entries != sorted(
        entries,
        key=lambda entry: (
            entry["canonical_launch_identity_sha256"],
            entry["crosswalk_model"]["canonical_model_sha256"],
        ),
    ):
        raise ImmutableArtifactError("canonical crosswalk entries are not canonically ordered")
    core = {
        "inventory_schema_version": 1,
        "entry_count": entry_count,
        "entries": entries,
    }
    expected_sha = derive_domain_sha256(
        "FIXTURE_CANONICAL_CROSSWALK_INVENTORY_V1", core
    )
    if value["inventory_sha256"] != expected_sha:
        raise ImmutableArtifactError("canonical crosswalk inventory SHA-256 mismatch")
    return {**core, "inventory_sha256": expected_sha}


def fixture_candidate_inventory_model(
    candidates: Iterable[CohortCandidateIdentity],
    *,
    crosswalk_inventory: dict[str, Any],
) -> dict[str, Any]:
    material = tuple(candidates)
    if any(not isinstance(candidate, CohortCandidateIdentity) for candidate in material):
        raise ImmutableArtifactError("cohort candidates have an invalid identity type")
    if not material:
        raise ImmutableArtifactError("cohort-candidate inventory must not be empty")
    require_unique_identities(
        material,
        identity=lambda candidate: candidate.candidate_id,
        label="cohort-candidate ID",
    )
    require_unique_identities(
        material,
        identity=lambda candidate: candidate.candidate_identity_sha256,
        label="cohort-candidate",
    )
    allowed_launches = {
        entry["canonical_launch_identity_sha256"]
        for entry in crosswalk_inventory["entries"]
    }
    if any(
        candidate.canonical_launch_identity_sha256 not in allowed_launches
        for candidate in material
    ):
        raise ImmutableArtifactError(
            "cohort candidate is absent from canonical crosswalk inventory"
        )
    ordered = sorted(material, key=lambda candidate: candidate.candidate_id)
    core = {
        "inventory_schema_version": 1,
        "selection_state": "FIXTURE_CANDIDATES_COMPLETE_NONAUTHORITATIVE",
        "candidate_count": len(ordered),
        "candidates": [candidate.to_model() for candidate in ordered],
    }
    return {
        **core,
        "inventory_sha256": derive_domain_sha256(
            "FIXTURE_COHORT_CANDIDATE_INVENTORY_V1", core
        ),
    }


def _parse_candidate_inventory(
    value: object,
    *,
    crosswalk_inventory: dict[str, Any],
) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
            "inventory_schema_version",
            "selection_state",
            "candidate_count",
            "candidates",
            "inventory_sha256",
        },
        "cohort-candidate inventory",
    )
    assert isinstance(value, dict)
    if type(value["inventory_schema_version"]) is not int or value[
        "inventory_schema_version"
    ] != 1:
        raise ImmutableArtifactError("candidate inventory schema version is unsupported")
    if value["selection_state"] != "FIXTURE_CANDIDATES_COMPLETE_NONAUTHORITATIVE":
        raise ImmutableArtifactError("candidate inventory selected a cohort")
    candidate_count = require_safe_nonnegative_integer(
        value["candidate_count"], "candidate_count"
    )
    raw_candidates = value["candidates"]
    if type(raw_candidates) is not list or not raw_candidates:
        raise ImmutableArtifactError("cohort candidates must be a nonempty array")
    candidates = [
        CohortCandidateIdentity.from_model(candidate) for candidate in raw_candidates
    ]
    if candidate_count != len(candidates):
        raise ImmutableArtifactError("candidate_count mismatch")
    require_unique_identities(
        candidates,
        identity=lambda candidate: candidate.candidate_id,
        label="cohort-candidate ID",
    )
    require_unique_identities(
        candidates,
        identity=lambda candidate: candidate.candidate_identity_sha256,
        label="cohort-candidate",
    )
    if candidates != sorted(candidates, key=lambda candidate: candidate.candidate_id):
        raise ImmutableArtifactError("cohort candidates are not canonically ordered")
    allowed_launches = {
        entry["canonical_launch_identity_sha256"]
        for entry in crosswalk_inventory["entries"]
    }
    if any(
        candidate.canonical_launch_identity_sha256 not in allowed_launches
        for candidate in candidates
    ):
        raise ImmutableArtifactError(
            "cohort candidate is absent from canonical crosswalk inventory"
        )
    core = {
        "inventory_schema_version": 1,
        "selection_state": "FIXTURE_CANDIDATES_COMPLETE_NONAUTHORITATIVE",
        "candidate_count": candidate_count,
        "candidates": [candidate.to_model() for candidate in candidates],
    }
    expected_sha = derive_domain_sha256(
        "FIXTURE_COHORT_CANDIDATE_INVENTORY_V1", core
    )
    if value["inventory_sha256"] != expected_sha:
        raise ImmutableArtifactError("cohort-candidate inventory SHA-256 mismatch")
    return {**core, "inventory_sha256": expected_sha}


def _expected_row_inventory_model(
    rows: Iterable[ExpectedPredictionRowIdentity],
    *,
    candidate_inventory: dict[str, Any],
) -> dict[str, Any]:
    material = tuple(rows)
    if any(not isinstance(row, ExpectedPredictionRowIdentity) for row in material):
        raise ImmutableArtifactError("expected rows have an invalid identity type")
    if not material:
        raise ImmutableArtifactError("expected row inventory must not be empty")
    _require_expected_row_coverage(material, candidate_inventory=candidate_inventory)
    ordered = sorted(material, key=lambda row: row.row_id)
    core = {
        "inventory_schema_version": 1,
        "row_count": len(ordered),
        "rows": [row.to_model() for row in ordered],
    }
    return {
        **core,
        "inventory_sha256": derive_domain_sha256(
            "FIXTURE_EXPECTED_PREDICTION_ROW_INVENTORY_V1", core
        ),
    }


def _parse_expected_row_inventory(
    value: object,
    *,
    candidate_inventory: dict[str, Any],
) -> dict[str, Any]:
    require_exact_keys(
        value,
        {"inventory_schema_version", "row_count", "rows", "inventory_sha256"},
        "expected row inventory",
    )
    assert isinstance(value, dict)
    if type(value["inventory_schema_version"]) is not int or value[
        "inventory_schema_version"
    ] != 1:
        raise ImmutableArtifactError("expected row inventory schema version is unsupported")
    row_count = require_safe_nonnegative_integer(value["row_count"], "row_count")
    raw_rows = value["rows"]
    if type(raw_rows) is not list or not raw_rows:
        raise ImmutableArtifactError("expected rows must be a nonempty array")
    rows = [ExpectedPredictionRowIdentity.from_model(row) for row in raw_rows]
    if row_count != len(rows):
        raise ImmutableArtifactError("expected row_count mismatch")
    _require_expected_row_coverage(rows, candidate_inventory=candidate_inventory)
    if rows != sorted(rows, key=lambda row: row.row_id):
        raise ImmutableArtifactError("expected rows are not canonically ordered")
    core = {
        "inventory_schema_version": 1,
        "row_count": row_count,
        "rows": [row.to_model() for row in rows],
    }
    expected_sha = derive_domain_sha256(
        "FIXTURE_EXPECTED_PREDICTION_ROW_INVENTORY_V1", core
    )
    if value["inventory_sha256"] != expected_sha:
        raise ImmutableArtifactError("expected row inventory SHA-256 mismatch")
    return {**core, "inventory_sha256": expected_sha}


def _require_expected_row_coverage(
    rows: Sequence[ExpectedPredictionRowIdentity],
    *,
    candidate_inventory: dict[str, Any],
) -> None:
    require_unique_identities(
        rows,
        identity=lambda row: row.row_id,
        label="expected row ID",
    )
    require_unique_identities(
        rows,
        identity=lambda row: row.expected_row_identity_sha256,
        label="expected row",
    )
    require_unique_identities(
        rows,
        identity=lambda row: row.cohort_candidate_identity_sha256,
        label="expected-row cohort candidate",
    )
    require_unique_identities(
        rows,
        identity=lambda row: row.simulator_output_row_sha256,
        label="simulator output row",
    )
    candidates = [
        CohortCandidateIdentity.from_model(candidate)
        for candidate in candidate_inventory["candidates"]
    ]
    by_identity = {
        candidate.candidate_identity_sha256: candidate for candidate in candidates
    }
    if {row.cohort_candidate_identity_sha256 for row in rows} != set(by_identity):
        raise ImmutableArtifactError("expected rows do not cover every cohort candidate")
    for row in rows:
        candidate = by_identity[row.cohort_candidate_identity_sha256]
        if row.row_id != candidate.candidate_id:
            raise ImmutableArtifactError("expected row ID does not match its candidate ID")
        if (
            row.canonical_launch_identity_sha256
            != candidate.canonical_launch_identity_sha256
        ):
            raise ImmutableArtifactError(
                "expected row canonical launch does not match its candidate"
            )


def _require_manifest_row_coverage(
    manifest: FixtureSimulatorOutputManifest,
    *,
    row_inventory: dict[str, Any],
) -> None:
    if not isinstance(manifest, FixtureSimulatorOutputManifest):
        raise ImmutableArtifactError("fixture output manifest type is invalid")
    expected = tuple(
        sorted(
            (
                (row["row_id"], row["simulator_output_row_sha256"])
                for row in row_inventory["rows"]
            ),
            key=lambda item: item[0],
        )
    )
    if manifest.entries != expected:
        raise ImmutableArtifactError(
            "simulator output manifest does not exactly cover expected rows"
        )


def _canonical_launch_identity_sha256(crosswalk: dict[str, Any]) -> str:
    try:
        selected = crosswalk["namespaces"]["canonical"]["selected_launch"]
    except (KeyError, TypeError) as exc:
        raise ImmutableArtifactError(
            "crosswalk canonical launch identity is unavailable"
        ) from exc
    if selected is None:
        raise ImmutableArtifactError("crosswalk canonical launch is not selected")
    launch = LaunchRecord.from_model(selected)
    if launch.identity.namespace is not LaunchNamespace.CANONICAL:
        raise ImmutableArtifactError("crosswalk canonical launch namespace mismatch")
    return launch.identity.qualified_launch_sha256


def _verify_schema_input(
    schemas: ReceiptValidationSchemas,
) -> RetainedSchemaInput:
    if not isinstance(schemas, ReceiptValidationSchemas):
        raise ImmutableArtifactError("receipt validation schemas are invalid")
    schema = schemas.prediction_seal
    verify_retained_schema(
        schema.schema_bytes,
        schema.schema_snapshot,
        expected_sha256=SCHEMA_SHA256,
        expected_id=SCHEMA_ID,
    )
    return schema
