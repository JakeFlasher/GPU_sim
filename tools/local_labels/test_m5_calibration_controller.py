from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest
from jsonschema import Draft202012Validator

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import m5_calibration_controller as m5


_COMMIT = "1" * 40
_SEED = bytes(range(32))
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_TRUST_REGISTRY_PATH = m5.TRUST_ANCHOR_REGISTRY_PATH_V1
_PRODUCTION_TRUST_REGISTRY_SHA256 = m5.TRUST_ANCHOR_REGISTRY_SHA256_V1


def _encode_point(point: tuple[int, int, int, int]) -> bytes:
    x, y, z, _t = point
    inverse = pow(z, m5._ED25519_Q - 2, m5._ED25519_Q)
    x = x * inverse % m5._ED25519_Q
    y = y * inverse % m5._ED25519_Q
    encoded = bytearray(y.to_bytes(32, "little"))
    encoded[31] |= (x & 1) << 7
    return bytes(encoded)


def _test_key(seed: bytes = _SEED) -> tuple[bytes, int, bytes]:
    digest = hashlib.sha512(seed).digest()
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    public = _encode_point(m5._ed25519_scalar_mult(m5._ED25519_BASE, scalar))
    return public, scalar, digest[32:]


def _sign(message: bytes, seed: bytes = _SEED) -> bytes:
    public, scalar, prefix = _test_key(seed)
    nonce = int.from_bytes(hashlib.sha512(prefix + message).digest(), "little") % m5._ED25519_L
    encoded_r = _encode_point(m5._ed25519_scalar_mult(m5._ED25519_BASE, nonce))
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public + message).digest(), "little"
    ) % m5._ED25519_L
    signature_scalar = (nonce + challenge * scalar) % m5._ED25519_L
    return encoded_r + signature_scalar.to_bytes(32, "little")


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_json(path: Path, value: Any) -> Path:
    return _write(path, immutable.pretty_json_bytes(value))


def _input(artifact_id: str, path: Path, media_type: str = "application/json") -> dict[str, Any]:
    return {"artifact_id": artifact_id, "path": str(path), "media_type": media_type}


def _identity(root: Path, identity_id: str, kind: str, version: str, payload: bytes, media_type: str) -> dict[str, Any]:
    path = _write(root / "identity-inputs" / identity_id, payload)
    return {
        "identity_id": identity_id,
        "identity_kind": kind,
        "version": version,
        "artifact_input": _input(f"{identity_id}-input", path, media_type),
    }


def _install_fixture_trust_registry(identities: list[dict[str, Any]]) -> None:
    by_id = {identity["identity_id"]: identity for identity in identities}

    def artifact_sha(identity_id: str) -> str:
        return immutable.sha256_file(
            Path(by_id[identity_id]["artifact_input"]["path"])
        )[0]

    authority_path = Path(
        by_id["m5-quantity-authority-v2"]["artifact_input"]["path"]
    )
    authority_raw = authority_path.read_bytes()
    authority = json.loads(authority_raw)
    design_inputs = {}
    design_input_witnesses = []
    for binding in authority["provenance"]["design_input_bindings"]:
        identity_id = binding["identity_id"]
        identity = by_id[identity_id]
        path = Path(identity["artifact_input"]["path"])
        payload = path.read_bytes()
        reference = {
            "artifact_id": f"{identity_id}-input",
            "relative_path": m5._artifact_storage_path(
                f"{identity_id}-input", immutable.sha256_bytes(payload)
            ),
            "sha256": immutable.sha256_bytes(payload),
            "size_bytes": len(payload),
            "media_type": identity["artifact_input"]["media_type"],
        }
        design_inputs[identity_id] = (reference, payload)
        design_input_witnesses.append(
            {
                "identity_id": identity_id,
                "sha256": reference["sha256"],
                "size_bytes": reference["size_bytes"],
                "media_type": reference["media_type"],
            }
        )
    audit, audit_payload = m5._build_ancestry_audit(
        authority=authority,
        authority_raw=authority_raw,
        design_inputs=design_inputs,
        source_registry_payload=m5.M5_SOURCE_REGISTRY_PATH_V1.read_bytes(),
        source_registry_schema_payload=(
            m5.M5_SOURCE_REGISTRY_SCHEMA_PATH_V1.read_bytes()
        ),
        audit_tool_payload=(
            m5.AUTHORITY_ANCESTRY_AUDIT_TOOL_PATH_V1.read_bytes()
        ),
        schema_payload=(
            m5.AUTHORITY_ANCESTRY_AUDIT_SCHEMA_PATH_V1.read_bytes()
        ),
    )
    audit_sha256 = immutable.sha256_bytes(audit_payload)
    audit_policy_sha256 = audit["audit_policy_sha256"]

    registry = {
        "schema_version": 1,
        "artifact_kind": "M5_TRUST_ANCHOR_REGISTRY",
        "entries": [
            {
                "anchor_id": "fixture-m5-authority",
                "authority_domain": "sm86_m5_rule1_quantity_authority_v2",
                "trust_anchor_identity_id": "m5-analysis-authority",
                "quantity_authority_identity_id": "m5-quantity-authority-v2",
                "algorithm": "Ed25519",
                "public_key_sha256": artifact_sha("m5-analysis-authority"),
                "authority_sha256": artifact_sha("m5-quantity-authority-v2"),
                "ancestry_identity_id": "m5-quantity-authority-ancestry",
                "ancestry_sha256": artifact_sha("m5-quantity-authority-ancestry"),
                "ancestry_audit_identity_id": (
                    "m5-quantity-authority-ancestry-audit"
                ),
                "ancestry_audit_sha256": audit_sha256,
                "audit_policy_sha256": audit_policy_sha256,
                "design_inputs": sorted(
                    design_input_witnesses, key=lambda item: item["identity_id"]
                ),
                "review_witnesses": [
                    {
                        "identity_id": "m5-quantity-authority-review",
                        "sha256": artifact_sha("m5-quantity-authority-review"),
                    }
                ],
                "operator_approval_identity_id": (
                    "m5-quantity-authority-operator-approval"
                ),
                "operator_approval_sha256": artifact_sha(
                    "m5-quantity-authority-operator-approval"
                ),
                "admission_basis": "EXPLICIT_REVIEW_AND_OPERATOR_ACCEPTANCE",
            }
        ],
    }
    payload = immutable.pretty_json_bytes(registry)
    _write(m5.TRUST_ANCHOR_REGISTRY_PATH_V1, payload)
    m5.TRUST_ANCHOR_REGISTRY_SHA256_V1 = immutable.sha256_bytes(payload)


def _resign_current_fixture_authority(
    request: dict[str, Any],
    *,
    seed: bytes = _SEED,
    install_registry: bool = True,
) -> None:
    by_id = {
        identity["identity_id"]: identity for identity in request["identities"]
    }
    authority_path = Path(
        by_id["m5-quantity-authority-v2"]["artifact_input"]["path"]
    )
    authority = json.loads(authority_path.read_bytes())
    unsigned = copy.deepcopy(authority)
    unsigned.pop("signature_ed25519")
    authority["signature_ed25519"] = _sign(
        m5._AUTHORITY_SIGNATURE_DOMAIN
        + immutable.canonical_json_bytes(unsigned),
        seed=seed,
    ).hex()
    authority_raw = immutable.pretty_json_bytes(authority)
    authority_path.write_bytes(authority_raw)
    authority_sha256 = immutable.sha256_bytes(authority_raw)
    for identity_id in (
        "m5-quantity-authority-ancestry",
        "m5-quantity-authority-review",
        "m5-quantity-authority-operator-approval",
    ):
        witness_path = Path(by_id[identity_id]["artifact_input"]["path"])
        witness = json.loads(witness_path.read_bytes())
        witness["authority_sha256"] = authority_sha256
        witness_path.write_bytes(immutable.pretty_json_bytes(witness))
    request["quantities"] = copy.deepcopy(authority["quantities"])
    request["family_execution_order"] = copy.deepcopy(authority["execution_order"])
    request["family_scientific_rank"] = copy.deepcopy(authority["scientific_rank"])
    if install_registry:
        _install_fixture_trust_registry(request["identities"])


def _resign_fixture_authority(
    request: dict[str, Any],
    quantities: list[dict[str, Any]],
    *,
    install_registry: bool = True,
) -> None:
    by_id = {
        identity["identity_id"]: identity for identity in request["identities"]
    }
    authority_path = Path(
        by_id["m5-quantity-authority-v2"]["artifact_input"]["path"]
    )
    authority = json.loads(authority_path.read_bytes())
    authority["quantities"] = copy.deepcopy(quantities)
    authority["consumer_binding_registry"] = [
        {
            "quantity_id": quantity["quantity_id"],
            "consumer_binding": quantity["consumer_binding"],
            "authorized_gap_kinds": list(quantity["authorized_gap_kinds"]),
            "research_gap_only": True,
        }
        for quantity in quantities
    ]
    design_input_ids = [
        binding["identity_id"]
        for binding in authority["provenance"]["design_input_bindings"]
    ]
    authority["provenance"]["scientific_field_sources"] = [
        {
            "field_pointer": pointer,
            "design_input_identity_ids": design_input_ids,
        }
        for pointer in (
            "/execution_order",
            "/scientific_rank",
            *(
                f"/quantities/{index}"
                for index in range(len(quantities))
            ),
        )
    ]
    authority_path.write_bytes(immutable.pretty_json_bytes(authority))
    _resign_current_fixture_authority(
        request, install_registry=install_registry
    )


def _plan_request(root: Path) -> dict[str, Any]:
    public, _scalar, _prefix = _test_key()
    quantities = [
        quantity
        for quantity in copy.deepcopy(
            m5._quantity_authority_model()["quantities"]
        )
        if quantity["quantity_id"] != "c.dram_copy_bandwidth"
    ]
    for quantity in quantities:
        quantity["meaning"] = f"Fixture-only synthetic meaning for {quantity['quantity_id']}"
        quantity["conditioning"]["variant"] = f"fixture.{quantity['quantity_id']}"
    fixture_execution_order = ["a", "c", "e", "b", "d", "f"]
    fixture_scientific_rank = ["e", "c", "a", "f", "d", "b"]
    synthetic_dependencies = {
        "a": [],
        "c": ["a"],
        "e": [],
        "b": ["c"],
        "d": ["a", "e"],
        "f": ["c", "e"],
    }
    for quantity in quantities:
        quantity["upstream_family_ids"] = (
            synthetic_dependencies[quantity["producer"]["family_id"]]
            if quantity["producer"]["kind"] == "PROBE_FAMILY"
            else []
        )
    execution_index = {
        family: index for index, family in enumerate(fixture_execution_order)
    }
    quantities.sort(
        key=lambda quantity: (
            execution_index.get(quantity["producer"].get("family_id"), 6),
            quantity["quantity_id"],
        )
    )
    design_input_id = "fixture-scientific-design"
    design_input_payload = (
        b"Synthetic offline fixture design. Every field is test-only and grants no "
        b"scientific, reviewer, operator, campaign, or hardware credit.\n"
    )
    design_input_binding = {
        "identity_id": design_input_id,
        "sha256": immutable.sha256_bytes(design_input_payload),
        "size_bytes": len(design_input_payload),
        "media_type": "text/markdown",
        "role": "SCIENTIFIC_DESIGN_INPUT",
    }
    scientific_field_sources = [
        {
            "field_pointer": pointer,
            "design_input_identity_ids": [design_input_id],
        }
        for pointer in (
            "/execution_order",
            "/scientific_rank",
            *(f"/quantities/{index}" for index in range(len(quantities))),
        )
    ]
    authority_unsigned = {
        "schema_version": 2,
        "artifact_kind": "M5_QUANTITY_AUTHORITY",
        "authority_id": "sm86-m5-quantity-authority-v2",
        "execution_order": fixture_execution_order,
        "scientific_rank": fixture_scientific_rank,
        "quantities": quantities,
        "consumer_binding_registry": [
            {
                "quantity_id": quantity["quantity_id"],
                "consumer_binding": quantity["consumer_binding"],
                "authorized_gap_kinds": list(quantity["authorized_gap_kinds"]),
                "research_gap_only": True,
            }
            for quantity in quantities
        ],
        "gap_mapping": {
            "mapping_version": 1,
            "typed_gap_kinds": sorted(m5.TYPED_GAPS),
            "consumer_rule": "RETAIN_AS_TYPED_RESEARCH_GAP",
            "ncu_selection_forbidden": True,
        },
        "provenance": {
            "ancestry_identity_id": "m5-quantity-authority-ancestry",
            "review_identity_ids": ["m5-quantity-authority-review"],
            "operator_approval_identity_id": "m5-quantity-authority-operator-approval",
            "trust_anchor_identity_id": "m5-analysis-authority",
            "target_free": True,
            "design_input_bindings": [design_input_binding],
            "scientific_field_sources": scientific_field_sources,
        },
    }
    authority = {
        **authority_unsigned,
        "signature_ed25519": _sign(
            m5._AUTHORITY_SIGNATURE_DOMAIN
            + immutable.canonical_json_bytes(authority_unsigned)
        ).hex(),
    }
    authority_raw = immutable.pretty_json_bytes(authority)
    authority_sha = immutable.sha256_bytes(authority_raw)
    ancestry = {
        "schema_version": 1,
        "artifact_kind": "M5_AUTHORITY_ANCESTRY",
        "authority_sha256": authority_sha,
        "target_free": True,
        "content_informing_ancestor_ids": [],
        "forbidden_source_classes": [
            "NCU_TARGET",
            "SIMULATOR_ERROR",
            "TARGET_BRACKET",
            "DERIVED_SELECTION",
        ],
    }
    review = {
        "schema_version": 1,
        "artifact_kind": "M5_AUTHORITY_BLIND_REVIEW",
        "authority_sha256": authority_sha,
        "reviewer_id": "reviewer-independent-01",
        "blind_review": True,
        "independent": True,
        "approved": True,
    }
    operator = {
        "schema_version": 1,
        "artifact_kind": "M5_AUTHORITY_OPERATOR_APPROVAL",
        "authority_sha256": authority_sha,
        "operator_id": "operator-test",
        "approved": True,
    }
    identities = [
        _identity(root, "m5-analysis-authority", "AUTHORITY", "ed25519-v1", public, "application/vnd.m5.ed25519-public-key"),
        _identity(root, "m5-quantity-authority-v2", "AUTHORITY", "v2", authority_raw, m5.QUANTITY_AUTHORITY_MEDIA_TYPE_V2),
        _identity(root, "m5-quantity-authority-ancestry", "AUTHORITY", "v1", immutable.pretty_json_bytes(ancestry), "application/json"),
        _identity(root, "m5-quantity-authority-review", "AUTHORITY", "v1", immutable.pretty_json_bytes(review), "application/json"),
        _identity(root, "m5-quantity-authority-operator-approval", "AUTHORITY", "v1", immutable.pretty_json_bytes(operator), "application/json"),
        _identity(root, design_input_id, "DESIGN_INPUT", "v1", design_input_payload, "text/markdown"),
        _identity(root, "source-main", "SOURCE", "v1", b"source\n", "text/x-cuda"),
        _identity(root, "harness-main", "HARNESS", "v1", b"harness\n", "text/x-python"),
        _identity(root, "build-main", "BUILD", "v1", b"build\n", "application/json"),
        _identity(root, "binary-main", "BINARY", "v1", b"binary\n", "application/octet-stream"),
    ]
    pchase_ids = [q["quantity_id"] for q in quantities if q["producer"]["kind"] == "PCHASE"]
    relationship = {
        "relationship_id": "build-rel-main",
        "source_identity_ids": ["source-main", "m5-pchase-source"],
        "build_identity_id": "build-main",
        "binary_identity_id": "binary-main",
    }
    rank = {
        family: index + 1
        for index, family in enumerate(fixture_scientific_rank)
    }
    families = [
        {
            "family_id": family,
            "execution_index": index,
            "scientific_rank": rank[family],
            "harness_identity_ids": ["harness-main"],
            "build_relationship_ids": ["build-rel-main"],
        }
        for index, family in enumerate(fixture_execution_order, start=1)
    ]
    return {
        "schema_version": 2,
        "request_kind": "M5_PLAN_REQUEST",
        "expected_source_commit": _COMMIT,
        "target": {"architecture": "sm_86", "chip": "GA102", "sku": "RTX 3090 Ti", "device_uuid_sha256": "2" * 64},
        "operating_state": {
            "state_id": "locked-sm86", "sm_clock_mhz": 1350, "memory_clock_mhz": 10501,
            "persistence_mode": True, "clock_locked": True, "cache_control": "all", "throttle_state": "idle",
            "driver_version": "610.43.02", "vbios_version": "test-vbios",
            "core_memory_relationship": {"kind": "LOCKED_FIXED_PAIR", "core_mhz": 1350, "memory_mhz": 10501, "reduced_core": 1350, "reduced_memory": 10501},
        },
        "pins": [{"pin_id": "source-authority", "value": _COMMIT, "authority_identity_id": "source-main", "authority_kind": "SOURCE"}],
        "identities": identities,
        "build_relationships": [relationship],
        "family_execution_order": fixture_execution_order,
        "family_scientific_rank": fixture_scientific_rank,
        "families": families,
        "quantities": quantities,
        "components": {
            "pchase": {"component_id": "pchase", "required": True, "harness_identity_ids": ["harness-main"], "build_relationship_ids": ["build-rel-main"], "quantity_ids": pchase_ids},
            "roster_control_sidecars": {
                "component_id": "roster_control_sidecars", "required": True, "expected_inventory_id": "control-inventory", "expected_binary_count": 1,
                "members": [{"binary_id": "binary-main", "binary_identity_id": "binary-main", "expected_sidecar_id": "control.calibration-main", "witness_slug": "calibration_main", "shape_point": "m5_locked"}],
            },
        },
        "policy": {
            "ncu_cross_check_only": True, "ncu_derivation_forbidden": True, "ncu_selection_forbidden": True,
            "ncu_tuning_forbidden": True, "ncu_transcription_forbidden": True, "simulator_fit_forbidden": True,
            "architecture_card_write_forbidden": True, "publication_mode": "ADDITIVE_EVIDENCE_ONLY",
        },
    }


def _control_record() -> dict[str, Any]:
    return {
        "pc": 0, "predicate": None, "opcode_text": "MOV R0, R0", "low_word": "0x0000000000000000", "high_word": "0x0000000000000000",
        "control": {"stall": 0, "yield_flag": 0, "write_barrier": 0, "read_barrier": 0, "wait_mask": 0, "reuse": 0}, "status": "decoded",
    }


def _sidecar(function_name: str, binary_name: str) -> dict[str, Any]:
    return {
        "arch": "sm_86", "functions": {function_name: [_control_record()]}, "skipped_arch_sections": [],
        "parser_contract": copy.deepcopy(m5.CONTROL_PARSER_CONTRACT), "toolkit_version": "V13.2.0", "binary": binary_name,
        "summary": {"functions": {function_name: {"instructions": 1, "decoded": 1, "unknown_encoding": 0, "stall_cycles_static_sum": 0}}, "total_decoded": 1, "total_unknown_encoding": 0},
    }


def _full_join_cell(member: Mapping[str, Any], function_name: str) -> dict[str, Any]:
    return {
        "identity": {"slug": member["slug"], "shape_point": member["shape_point"], "binary_sha256": member["binary_sha256"]},
        "first_launch": "kernel_1", "vpc_low32_mismatches": 0, "n_instruction_records": 1,
        "functions": [{
            "function_unique_id": 1, "kernel_name": f"{function_name}___0", "sass_function": function_name, "status": "joined",
            "n_traced_pcs": 1, "n_joined": 1, "n_decoded": 1, "missing_pcs": [],
            "sample": {"pc": 0, "opcode_text": "MOV R0, R0", "control": _control_record()["control"]},
        }],
        "coverage": {"n_traced_pcs": 1, "n_joined": 1, "n_decoded": 1, "full_join": True},
    }


def _typed_join_cell(member: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": {"slug": member["slug"], "shape_point": member["shape_point"], "binary_sha256": member["binary_sha256"]},
        "status": "typed", "typed_reason": "binary_or_trace_unresolved",
    }


class CampaignBuilder:
    def __init__(self, root: Path, run: Mapping[str, Any]) -> None:
        self.root = root
        self.run = run
        self.counter = 0
        self.repetition_counter = 0

    def authenticated(self, artifact_id: str, payload: bytes, role: str, context_id: str, media_type: str = "application/json") -> tuple[dict[str, Any], dict[str, Any], Path]:
        self.counter += 1
        artifact_path = _write(self.root / "analysis-inputs" / f"artifact-{self.counter:04d}", payload)
        digest = immutable.sha256_bytes(payload)
        reference = m5._artifact_reference(artifact_id, digest, len(payload), media_type)
        unsigned = {
            "schema_version": 1, "artifact_kind": "M5_AUTHORITY_RECEIPT", "authority_identity_id": "m5-analysis-authority",
            "run_model_seal_sha256": self.run["model_seal_sha256"], "source_commit": self.run["source_state"]["source_commit"],
            "artifact_id": artifact_id, "artifact_sha256": digest, "artifact_size_bytes": len(payload), "artifact_media_type": media_type,
            "artifact_role": role, "context_id": context_id,
        }
        signature = _sign(m5._RECEIPT_DOMAIN + immutable.canonical_json_bytes(unsigned))
        receipt = {**unsigned, "signature_ed25519": signature.hex()}
        receipt_id = f"receipt.{self.counter:04d}.{hashlib.sha256(context_id.encode()).hexdigest()[:12]}"
        receipt_path = _write_json(self.root / "analysis-inputs" / f"receipt-{self.counter:04d}.json", receipt)
        return {
            "artifact_input": _input(artifact_id, artifact_path, media_type),
            "receipt_input": _input(receipt_id, receipt_path),
        }, reference, artifact_path

    def repetitions(self) -> list[str]:
        result = []
        for _ in range(3):
            self.repetition_counter += 1
            result.append(f"rep-{self.repetition_counter:04d}")
        return result

    def build(self) -> tuple[dict[str, Any], dict[str, Path]]:
        quantity_results = []
        repetition_ids: list[str] = []
        producer_paths: dict[str, Path] = {}
        decision_hashes_by_family: dict[str, list[str]] = {}
        for quantity in self.run["quantities"]:
            quantity_id = quantity["quantity_id"]
            evidence: dict[str, list[dict[str, Any]]] = {}
            evidence_ids: dict[str, str] = {}
            for role in m5.EVIDENCE_ROLES:
                artifact_id = f"evidence.{quantity_id}.{role}"
                evidence_ids[role] = artifact_id
                payload = immutable.pretty_json_bytes({"artifact_kind": "M5_TEST_EVIDENCE", "quantity_id": quantity_id, "role": role})
                auth, _reference, _path = self.authenticated(artifact_id, payload, f"EVIDENCE_{role.upper()}", f"{quantity_id}.{role}.0")
                evidence[role] = [auth]
            readout_inputs = []
            producer_hashes = []
            for contract in quantity["readout_contracts"]:
                repetitions = self.repetitions()
                repetition_ids.extend(repetitions)
                probability = quantity["value_domain"]["kind"] == "PROBABILITY"
                producer_family = (
                    quantity["producer"]["family_id"]
                    if quantity["producer"]["kind"] == "PROBE_FAMILY"
                    else "pchase"
                )
                upstream_decision_sha256s = m5._expected_upstream_decision_sha256s(
                    decision_hashes_by_family, quantity
                )
                capture_started_unix_us = 100_000 + self.repetition_counter * 100
                capture_session_id = f"session.{contract['readout_id']}"
                dependency_contract = m5._quantity_dependency_contract(quantity)
                lineage_id = m5._expected_source_lineage_id(
                    self.run,
                    quantity,
                    contract,
                    harness_identity_id="harness-main",
                    build_relationship_id="build-rel-main",
                    binary_identity_id="binary-main",
                    capture_session_id=capture_session_id,
                )
                observations = [
                    {
                        "repetition_id": repetition,
                        "launch_id": f"launch-{repetition}",
                        "observed_unix_us": capture_started_unix_us + index + 1,
                        "numerator": 1 if probability else 10,
                        "denominator": 2 if probability else 1,
                    }
                    for index, repetition in enumerate(repetitions)
                ]
                raw_sample = {
                    "schema_version": 1,
                    "artifact_kind": "M5_RAW_SAMPLE_SERIES",
                    "run_model_seal_sha256": self.run["model_seal_sha256"],
                    "quantity_id": quantity_id,
                    "readout_id": contract["readout_id"],
                    "method_class": contract["method_class"],
                    "instrument": contract["instrument"],
                    "source_lineage_id": lineage_id,
                    "capture_session_id": capture_session_id,
                    "capture_started_unix_us": capture_started_unix_us,
                    "device_uuid_sha256": self.run["target"]["device_uuid_sha256"],
                    "harness_identity_id": "harness-main",
                    "build_relationship_id": "build-rel-main",
                    "binary_identity_id": "binary-main",
                    "upstream_decision_sha256s": upstream_decision_sha256s,
                    "dependency_contract": dependency_contract,
                    "observations": observations,
                    "ncu_ancestor_artifact_ids": [],
                    "simulator_ancestor_artifact_ids": [],
                }
                raw_id = f"raw.{contract['readout_id']}"
                raw_auth, raw_reference, raw_path = self.authenticated(
                    raw_id,
                    immutable.pretty_json_bytes(raw_sample),
                    "RAW_SAMPLE_SERIES",
                    contract["readout_id"],
                )
                producer = {
                    "schema_version": 1, "artifact_kind": "M5_READOUT_PRODUCER", "run_model_seal_sha256": self.run["model_seal_sha256"],
                    "quantity_id": quantity_id, "readout_id": contract["readout_id"], "method_class": contract["method_class"],
                    "instrument": contract["instrument"], "derivation_identity_id": contract["derivation_identity_id"],
                    "derivation_operation": contract["derivation_operation"], "unit": quantity["unit"], "input_class": contract["instrument"],
                    "sample_operation": contract["sample_operation"], "raw_sample_artifact_sha256": raw_reference["sha256"],
                    "source_lineage_id": lineage_id,
                    "evidence_artifact_ids": [evidence_ids[role] for role in contract["required_evidence_roles"]],
                    "ncu_artifact_ids": [], "simulator_artifact_ids": [],
                }
                payload = immutable.pretty_json_bytes(producer)
                producer_id = f"producer.{contract['readout_id']}"
                producer_auth, reference, path = self.authenticated(producer_id, payload, "READOUT_PRODUCER", contract["readout_id"])
                producer_paths[contract["readout_id"]] = path
                producer_paths[f"raw.{contract['readout_id']}"] = raw_path
                readout_inputs.append({"raw_sample": raw_auth, "producer": producer_auth})
                producer_hashes.append(reference["sha256"])
            decision = {
                "schema_version": 1, "artifact_kind": "M5_QUANTITY_DECISION", "run_model_seal_sha256": self.run["model_seal_sha256"],
                "quantity_id": quantity_id, "readout_producer_sha256s": producer_hashes, "outcome": {"kind": "MEASURED"},
                "ncu_artifact_ids": [], "simulator_artifact_ids": [],
            }
            decision_auth, decision_reference, _path = self.authenticated(f"decision.{quantity_id}", immutable.pretty_json_bytes(decision), "QUANTITY_DECISION", quantity_id)
            decision_hashes_by_family.setdefault(producer_family, []).append(decision_reference["sha256"])
            quantity_results.append({"quantity_id": quantity_id, "evidence": evidence, "readout_inputs": readout_inputs, "decision": decision_auth, "cross_checks": []})

        def state(observation_id: str, phase: str, repetition_id: str | None) -> dict[str, Any]:
            if phase == "M0":
                observed_unix_us = 0
            elif repetition_id is None:
                observed_unix_us = 100 if phase == "PRE" else 10_000_000
            else:
                ordinal = int(repetition_id.rsplit("-", 1)[1])
                observed_unix_us = 500 + ordinal if phase == "PRE" else 2_000_000 + ordinal
            return {
                "observation_id": observation_id, "observation_phase": phase, "repetition_id": repetition_id,
                "observed_unix_us": observed_unix_us,
                "sm_clock_mhz": 1350, "memory_clock_mhz": 10501, "persistence_mode": True, "clock_locked": True,
                "cache_control": "all", "driver_version": "610.43.02", "nvcc_release": "13.3",
                "ncu_version": "2026.1.0", "ncu_replay_mode": "kernel",
                "device_uuid_sha256": self.run["target"]["device_uuid_sha256"], "throttle_reasons": [],
            }

        protocol = {
            "schema_version": 1, "artifact_kind": "M5_PROTOCOL_STATE_WITNESS", "run_model_seal_sha256": self.run["model_seal_sha256"],
            "source_commit": self.run["source_state"]["source_commit"], "protocol_identity_id": "m5-campaign-protocol",
            "m0": state("obs-m0", "M0", None), "global_pre": state("obs-global-pre", "PRE", None), "global_post": state("obs-global-post", "POST", None),
            "repetitions": [{"repetition_id": repetition, "pre": state(f"obs-pre-{repetition}", "PRE", repetition), "post": state(f"obs-post-{repetition}", "POST", repetition)} for repetition in repetition_ids],
        }
        protocol_auth, _ref, protocol_path = self.authenticated("m5-protocol-state-witness", immutable.pretty_json_bytes(protocol), "PROTOCOL_STATE_WITNESS", "protocol-state")

        def control_group(members: list[Mapping[str, Any]], group_id: str, witness_id: str, complete_first_only: bool) -> tuple[dict[str, Any], Path]:
            cells = []
            entries = []
            full_count = 0
            for index, member in enumerate(members):
                function_name = f"function_{group_id.lower()}_{index}"
                sidecar_auth, _reference, _path = self.authenticated(
                    member["expected_sidecar_id"], immutable.pretty_json_bytes(_sidecar(function_name, member["cell_id"])), "CONTROL_SIDECAR", member["cell_id"]
                )
                entries.append({"cell_id": member["cell_id"], "expected_sidecar_id": member["expected_sidecar_id"], "sidecar": sidecar_auth})
                complete = index == 0 if complete_first_only else True
                if complete:
                    cells.append(_full_join_cell(member, function_name))
                    full_count += 1
                else:
                    cells.append(_typed_join_cell(member))
            witness = {
                "schema_version": 1, "artifact_kind": "sass_control_pc_join_witness", "arch": "sm_86",
                "source_commit": self.run["source_state"]["source_commit"], "n_cells": len(cells), "n_full_join": full_count, "cells": cells,
            }
            witness_auth, _reference, path = self.authenticated(witness_id, immutable.pretty_json_bytes(witness), "CONTROL_PC_JOIN_WITNESS", group_id)
            return {"pc_join_witness": witness_auth, "entries": entries}, path

        canonical_members = self.run["components"]["canonical_runnable_roster"]["members"]
        calibration_members = self.run["components"]["roster_control_sidecars"]["calibration_members"]
        canonical_controls, canonical_witness_path = control_group(canonical_members, "CANONICAL_ROSTER", "canonical-pc-join-witness", True)
        calibration_controls, calibration_witness_path = control_group(calibration_members, "CALIBRATION_BINARIES", "calibration-pc-join-witness", False)
        request = {
            "schema_version": 2, "request_kind": "M5_ANALYSIS_REQUEST", "run_model_seal_sha256": self.run["model_seal_sha256"],
            "protocol_state_witness": protocol_auth, "quantity_results": quantity_results,
            "roster_control_sidecars": {"canonical": canonical_controls, "calibration": calibration_controls},
        }
        return request, {"protocol": protocol_path, "canonical_witness": canonical_witness_path, "calibration_witness": calibration_witness_path, **producer_paths}


@pytest.fixture(scope="module", autouse=True)
def _stable_source(tmp_path_factory: pytest.TempPathFactory) -> None:
    monkeypatch = pytest.MonkeyPatch()
    trust_root = tmp_path_factory.mktemp("m5-trust-registry")
    monkeypatch.setattr(
        m5,
        "TRUST_ANCHOR_REGISTRY_PATH_V1",
        trust_root / "m5_trust_anchors.json",
    )
    monkeypatch.setattr(m5, "TRUST_ANCHOR_REGISTRY_SHA256_V1", "0" * 64)
    bootstrap_request = _plan_request(trust_root / "bootstrap")
    _install_fixture_trust_registry(bootstrap_request["identities"])
    monkeypatch.setattr(m5, "_observe_git_source", lambda: (_COMMIT, b""))
    yield
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def _clean_fixture_trust_registry(tmp_path: Path) -> None:
    bootstrap_request = _plan_request(tmp_path / "trust-bootstrap")
    _install_fixture_trust_registry(bootstrap_request["identities"])


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("m5-campaign")
    request_path = _write_json(root / "plan-request.json", _plan_request(root))
    run_bundle = m5.plan(request_path, root / "run-publications")
    run = json.loads((run_bundle / "RUN.json").read_text())
    builder = CampaignBuilder(root, run)
    analysis_request, input_paths = builder.build()
    analysis_path = _write_json(root / "analysis-request.json", analysis_request)
    final_bundle = m5.analyze(run_bundle, analysis_path, root / "result-publications")
    public, _scalar, _prefix = _test_key()
    return {
        "root": root, "run_bundle": run_bundle, "run": run, "analysis_path": analysis_path,
        "analysis_request": analysis_request, "input_paths": input_paths, "bundle": final_bundle,
        "authority_sha256": immutable.sha256_bytes(public),
    }


def _bundle_models(campaign: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = campaign["bundle"]
    values = [json.loads((bundle / name).read_text()) for name in ("RUN.json", "RESULT.json", "DONE.json")]
    return values[0], values[1], values[2]


def test_non_31_plan_analyze_relocate_and_verify(
    campaign: Mapping[str, Any], tmp_path: Path
) -> None:
    run, result, done = _bundle_models(campaign)
    assert run["family_execution_order"] == ["a", "c", "e", "b", "d", "f"]
    assert run["family_scientific_rank"] == ["e", "c", "a", "f", "d", "b"]
    assert run["counts"]["quantity_count"] == 30
    audit_identity = next(
        identity
        for identity in run["identities"]
        if identity["identity_id"] == m5.AUTHORITY_ANCESTRY_AUDIT_ID_V1
    )
    assert audit_identity["identity_kind"] == "AUDIT"
    assert run["provenance"]["ancestry_audit_artifact"] == audit_identity["artifact"]
    assert len(run["provenance"]["design_input_artifacts"]) == 1
    audit_model = json.loads(
        (campaign["run_bundle"] / audit_identity["artifact"]["relative_path"]).read_bytes()
    )
    assert audit_model["mechanical_non_revival_only"] is True
    assert audit_model["semantic_review_required"] is True
    assert audit_model["external_credit"] == "NONE"
    assert run["counts"]["canonical_runnable_cell_count"] == 122
    assert result["counts"]["measured_quantity_count"] == 30
    assert result["counts"]["raw_sample_series_count"] == 60
    assert result["counts"]["authenticated_artifact_count"] == 396
    assert result["counts"]["authority_receipt_count"] == 396
    assert result["counts"]["canonical_control_sidecar_count"] == 122
    assert result["components"]["pchase"]["property_count"] == 14
    assert done["counts"]["required_quantity_count"] == 30
    assert done["counts"]["result_quantity_count"] == 30
    assert done["counts"]["raw_sample_series_count"] == 60
    assert done["counts"]["authenticated_artifact_count"] == 396
    assert done["counts"]["authority_receipt_count"] == 396
    assert done["counts"]["authority_receipt_count"] == len(
        result["authenticity"]["entries"]
    )
    relocated = tmp_path / "relocated"
    shutil.copytree(campaign["bundle"], relocated)
    verified = m5.verify(relocated, campaign["authority_sha256"])
    assert verified["model_seal_sha256"] == done["model_seal_sha256"]


def test_authority_declared_dependencies_drive_result(
    campaign: Mapping[str, Any],
) -> None:
    run, result, _done = _bundle_models(campaign)
    expected = {
        "a": [],
        "c": ["a"],
        "e": [],
        "b": ["c"],
        "d": ["a", "e"],
        "f": ["c", "e"],
    }
    decision_hashes_by_family: dict[str, list[str]] = {}
    for quantity, row in zip(
        run["quantities"], result["quantity_results"], strict=True
    ):
        if quantity["producer"]["kind"] == "PROBE_FAMILY":
            producer_family = quantity["producer"]["family_id"]
            assert quantity["upstream_family_ids"] == expected[producer_family]
        else:
            producer_family = "pchase"
            assert quantity["upstream_family_ids"] == []
        contract = (
            "NONE"
            if not quantity["upstream_family_ids"]
            else "UPSTREAM_FAMILIES:"
            + ",".join(quantity["upstream_family_ids"])
        )
        expected_hashes = sorted(
            digest
            for family_id in quantity["upstream_family_ids"]
            for digest in decision_hashes_by_family[family_id]
        )
        assert all(
            readout["dependency_contract"] == contract
            and readout["upstream_decision_sha256s"] == expected_hashes
            for readout in row["readouts"]
        )
        decision_hashes_by_family.setdefault(producer_family, []).append(
            row["decision_artifact"]["sha256"]
        )


def test_post_signature_upstream_dependency_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    request = _plan_request(tmp_path)
    authority_identity = next(
        identity
        for identity in request["identities"]
        if identity["identity_id"] == "m5-quantity-authority-v2"
    )
    authority_path = Path(authority_identity["artifact_input"]["path"])
    authority = json.loads(authority_path.read_bytes())
    quantity = next(
        row
        for row in authority["quantities"]
        if row["producer"] == {"kind": "PROBE_FAMILY", "family_id": "c"}
    )
    quantity["upstream_family_ids"] = []
    authority_path.write_bytes(immutable.pretty_json_bytes(authority))
    request_path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(m5.M5ContractError, match="signature"):
        m5.plan(request_path, tmp_path / "publications")


def test_v2_authority_controls_method_names_groups_and_bounds(tmp_path: Path) -> None:
    request = _plan_request(tmp_path)
    quantities = copy.deepcopy(request["quantities"])
    probe_f = next(
        quantity
        for quantity in quantities
        if quantity["producer"] == {"kind": "PROBE_FAMILY", "family_id": "f"}
    )
    probe_f["quantity_group"] = "externally-declared-group"
    probe_f["agreement_limit_fraction"] = 0.2
    for index, readout in enumerate(probe_f["readout_contracts"]):
        readout["method_class"] = f"EXTERNAL_TARGET_FREE_METHOD_{index}"
        readout["instrument"] = "EXTERNAL_TARGET_FREE_INSTRUMENT"
        readout["method_contract"]["confounder_set"] = [
            f"external_confounder_{index}"
        ]
    _resign_fixture_authority(request, quantities)
    request_path = _write_json(tmp_path / "request.json", request)

    bundle = m5.plan(request_path, tmp_path / "publications")
    run = json.loads((bundle / "RUN.json").read_bytes())
    observed = next(
        quantity
        for quantity in run["quantities"]
        if quantity["quantity_id"] == probe_f["quantity_id"]
    )
    assert observed == probe_f


def test_retained_run_audit_is_recomputed_from_bound_inputs(
    campaign: Mapping[str, Any],
) -> None:
    run = copy.deepcopy(campaign["run"])
    audit_identity = next(
        identity
        for identity in run["identities"]
        if identity["identity_id"] == m5.AUTHORITY_ANCESTRY_AUDIT_ID_V1
    )
    audit_model = json.loads(
        (
            campaign["run_bundle"]
            / audit_identity["artifact"]["relative_path"]
        ).read_bytes()
    )
    audit_model["scientific_projection_sha256"] = "0" * 64
    forged_payload = immutable.pretty_json_bytes(audit_model)
    forged_reference = copy.deepcopy(audit_identity["artifact"])
    forged_reference["sha256"] = immutable.sha256_bytes(forged_payload)
    forged_reference["size_bytes"] = len(forged_payload)
    forged_reference["relative_path"] = m5._artifact_storage_path(
        forged_reference["artifact_id"], forged_reference["sha256"]
    )
    audit_identity["artifact"] = forged_reference
    run["provenance"]["ancestry_audit_artifact"] = forged_reference
    run.pop("model_seal_sha256")
    run = immutable.seal_model(run)
    run_schema_identity = next(
        identity
        for identity in run["identities"]
        if identity["identity_id"] == "m5-run-schema"
    )
    run_schema = json.loads(
        (
            campaign["run_bundle"]
            / run_schema_identity["artifact"]["relative_path"]
        ).read_bytes()
    )

    def loader(reference: Mapping[str, Any]) -> bytes:
        if reference == forged_reference:
            return forged_payload
        return immutable.read_relative(
            campaign["run_bundle"], reference["relative_path"]
        )

    with pytest.raises(m5.M5ContractError, match="retained audit differs"):
        m5._validate_run_semantics(run, run_schema, loader)


def test_run_bundle_verifier_recomputes_retained_audit(
    campaign: Mapping[str, Any], tmp_path: Path
) -> None:
    relocated = tmp_path / "tampered-run-bundle"
    shutil.copytree(campaign["run_bundle"], relocated)
    os.chmod(relocated / "artifacts", 0o700)
    os.chmod(relocated / "RUN.json", 0o600)
    run = json.loads((relocated / "RUN.json").read_bytes())
    audit_identity = next(
        identity
        for identity in run["identities"]
        if identity["identity_id"] == m5.AUTHORITY_ANCESTRY_AUDIT_ID_V1
    )
    audit_model = json.loads(
        (relocated / audit_identity["artifact"]["relative_path"]).read_bytes()
    )
    audit_model["scientific_projection_sha256"] = "0" * 64
    forged_payload = immutable.pretty_json_bytes(audit_model)
    forged_reference = copy.deepcopy(audit_identity["artifact"])
    forged_reference["sha256"] = immutable.sha256_bytes(forged_payload)
    forged_reference["size_bytes"] = len(forged_payload)
    forged_reference["relative_path"] = m5._artifact_storage_path(
        forged_reference["artifact_id"], forged_reference["sha256"]
    )
    _write(relocated / forged_reference["relative_path"], forged_payload)
    audit_identity["artifact"] = forged_reference
    run["provenance"]["ancestry_audit_artifact"] = forged_reference
    run.pop("model_seal_sha256")
    _write_json(relocated / "RUN.json", immutable.seal_model(run))

    with pytest.raises(m5.M5ContractError, match="retained audit differs"):
        m5._verify_run_bundle(relocated)


def test_run_audit_recomputation_rejects_nonexecuting_retained_tool(
    campaign: Mapping[str, Any],
) -> None:
    run = campaign["run"]
    tool_identity = next(
        identity
        for identity in run["identities"]
        if identity["identity_id"] == "m5-ancestry-audit-tool"
    )
    run_schema_identity = next(
        identity
        for identity in run["identities"]
        if identity["identity_id"] == "m5-run-schema"
    )
    run_schema = json.loads(
        (
            campaign["run_bundle"]
            / run_schema_identity["artifact"]["relative_path"]
        ).read_bytes()
    )
    retained_tool = (
        campaign["run_bundle"] / tool_identity["artifact"]["relative_path"]
    ).read_bytes()

    def loader(reference: Mapping[str, Any]) -> bytes:
        if reference == tool_identity["artifact"]:
            return retained_tool + b"\n"
        return immutable.read_relative(
            campaign["run_bundle"], reference["relative_path"]
        )

    with pytest.raises(m5.M5ContractError, match="executing tool differs"):
        m5._validate_run_semantics(run, run_schema, loader)


def test_verify_requires_out_of_band_authority(campaign: Mapping[str, Any]) -> None:
    with pytest.raises(m5.M5ContractError, match="out-of-band trust anchor"):
        m5.verify(campaign["bundle"], "0" * 64)


def test_closed_roster_normalizes_uppercase_without_alias(campaign: Mapping[str, Any]) -> None:
    members = campaign["run"]["components"]["canonical_runnable_roster"]["members"]
    assert len(members) == 122
    assert len({member["cell_id"] for member in members}) == 122
    assert all(m5._SAFE_ID.fullmatch(member["cell_id"]) for member in members)
    uppercase = [member for member in members if any(character.isupper() for character in member["slug"])]
    assert uppercase
    assert all(member["cell_id"] == member["cell_id"].lower() for member in uppercase)


def test_result_rederives_readouts_not_caller_values(campaign: Mapping[str, Any]) -> None:
    run, result, _done = _bundle_models(campaign)
    quantity = run["quantities"][0]
    readout = copy.deepcopy(result["quantity_results"][0]["readouts"][0])
    readout["value"] += 1
    loader = m5._DeclaredSnapshotLoader(root=campaign["bundle"])
    try:
        with pytest.raises(m5.M5ContractError, match="not rederived"):
            m5._rederive_readout(run, quantity, quantity["readout_contracts"][0], readout, loader)
    finally:
        loader.close()


def test_raw_sample_repetition_cardinality_is_exact(campaign: Mapping[str, Any]) -> None:
    run, result, _done = _bundle_models(campaign)
    quantity = run["quantities"][0]
    readout = result["quantity_results"][0]["readouts"][0]
    raw_sample = json.loads(
        (campaign["bundle"] / readout["raw_sample_artifact"]["relative_path"]).read_text()
    )
    raw_sample["observations"][1]["repetition_id"] = raw_sample["observations"][0][
        "repetition_id"
    ]
    modified = immutable.pretty_json_bytes(raw_sample)

    def loader(reference: Mapping[str, Any]) -> bytes:
        if reference["artifact_id"] == readout["raw_sample_artifact"]["artifact_id"]:
            return modified
        return immutable.read_relative(campaign["bundle"], reference["relative_path"])

    with pytest.raises(m5.M5ContractError, match="duplicates a repetition"):
        m5._rederive_readout(run, quantity, quantity["readout_contracts"][0], readout, loader)


@pytest.mark.parametrize(
    ("quantity_id", "value", "match"),
    [
        ("pchase.l1_capacity_bytes", -1, "positive"),
        ("pchase.l1_capacity_bytes", 1.5, "integral"),
        ("a.l2_pair_fill_probability", 1.1, r"\[0,1\]"),
        ("e.l2_slice_hash_structure", -0.1, "nonnegative"),
    ],
)
def test_quantity_domains_are_closed(quantity_id: str, value: float, match: str) -> None:
    quantity = next(row for row in m5._quantity_authority_model()["quantities"] if row["quantity_id"] == quantity_id)
    with pytest.raises(m5.M5ContractError, match=match):
        m5._quantity_number(quantity, value, quantity_id)


def test_protocol_covers_every_readout_repetition(campaign: Mapping[str, Any]) -> None:
    _run, result, _done = _bundle_models(campaign)
    expected = sum(readout["sample_count"] for row in result["quantity_results"] for readout in row["readouts"])
    protocol = result["components"]["protocol_state"]
    assert protocol["repetition_count"] == expected == 180
    assert protocol["all_state_joins_passed"] is True


def test_control_inputs_are_real_standard_sidecars_and_pc_witnesses(campaign: Mapping[str, Any]) -> None:
    _run, result, _done = _bundle_models(campaign)
    controls = result["components"]["roster_control_sidecars"]
    assert controls["observed_canonical_cell_count"] == 122
    assert controls["observed_calibration_binary_count"] == 1
    assert controls["entries"][0]["status"] == "COMPLETE"
    assert any(entry["status"] == "TYPED" for entry in controls["entries"][1:122])
    sidecar = json.loads((campaign["bundle"] / controls["entries"][0]["sidecar"]["relative_path"]).read_text())
    assert set(sidecar) == {"arch", "functions", "skipped_arch_sections", "parser_contract", "toolkit_version", "binary", "summary"}


def test_done_inventory_contains_receipts_producers_and_decisions(campaign: Mapping[str, Any]) -> None:
    _run, result, done = _bundle_models(campaign)
    bindings = {binding for row in done["artifact_inventory"] for binding in row["bindings"]}
    assert {"AUTHORITY_RECEIPT", "READOUT_PRODUCER", "QUANTITY_DECISION", "PCHASE_FRESH_CARD"}.issubset(bindings)
    receipt_ids = {row["receipt"]["artifact_id"] for row in result["authenticity"]["entries"]}
    inventory_ids = {row["artifact_id"] for row in done["artifact_inventory"]}
    assert receipt_ids.issubset(inventory_ids)


def test_artifact_relative_path_is_canonical() -> None:
    reference = m5._artifact_reference("artifact", "a" * 64, 1, "text/plain")
    reference["relative_path"] = "artifacts/bad/artifact"
    with pytest.raises(m5.M5ContractError, match="canonical content-addressed"):
        m5._collect_artifact({}, reference, "artifact")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b'{"x":1,"x":2}', "duplicate"),
        (b'{"x":NaN}', "non-finite"),
        (b'{"x":9007199254740992}', "safe range"),
        (b'{"x":9007199254740993.0}', "float exceeds"),
        (b'{"x":"\\ud800"}', "surrogate"),
    ],
)
def test_strict_json_rejects_ambiguous_values(payload: bytes, match: str) -> None:
    with pytest.raises(immutable.ImmutableArtifactError, match=match):
        immutable.strict_json_loads(payload)


def test_strict_json_types_deep_nesting() -> None:
    payload = ("[" * 3000 + "0" + "]" * 3000).encode()
    with pytest.raises(immutable.ImmutableArtifactError, match="invalid|nesting"):
        immutable.strict_json_loads(payload)


def test_invalid_ed25519_authority_is_rejected(tmp_path: Path) -> None:
    request = _plan_request(tmp_path)
    authority = next(identity for identity in request["identities"] if identity["identity_id"] == "m5-analysis-authority")
    Path(authority["artifact_input"]["path"]).write_bytes(bytes(32))
    path = _write_json(tmp_path / "request.json", request)
    with pytest.raises(m5.M5ContractError, match="prime-order"):
        m5.plan(path, tmp_path / "out")


def test_unregistered_analysis_authority_is_rejected(tmp_path: Path) -> None:
    request = _plan_request(tmp_path)
    authority = next(
        identity
        for identity in request["identities"]
        if identity["identity_id"] == "m5-analysis-authority"
    )
    alternate_seed = bytes(reversed(_SEED))
    alternate_public, _scalar, _prefix = _test_key(alternate_seed)
    Path(authority["artifact_input"]["path"]).write_bytes(alternate_public)
    _resign_current_fixture_authority(
        request, seed=alternate_seed, install_registry=False
    )
    path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(m5.M5ContractError, match="TRUST_ANCHOR_UNADMITTED"):
        m5.plan(path, tmp_path / "out")


def test_empty_public_registry_admits_no_fixture_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _plan_request(tmp_path)
    monkeypatch.setattr(
        m5, "TRUST_ANCHOR_REGISTRY_PATH_V1", _PRODUCTION_TRUST_REGISTRY_PATH
    )
    monkeypatch.setattr(
        m5,
        "TRUST_ANCHOR_REGISTRY_SHA256_V1",
        _PRODUCTION_TRUST_REGISTRY_SHA256,
    )
    path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(m5.M5ContractError, match="TRUST_ANCHOR_UNADMITTED"):
        m5.plan(path, tmp_path / "out")


def test_v1_quantity_authority_is_quarantined(tmp_path: Path) -> None:
    request = _plan_request(tmp_path)
    legacy = m5._quantity_authority_model()
    request["identities"].append(
        _identity(
            tmp_path,
            "m5-quantity-authority",
            "AUTHORITY",
            "v1",
            immutable.pretty_json_bytes(legacy),
            "application/json",
        )
    )
    path = _write_json(tmp_path / "request.json", request)
    with pytest.raises(m5.M5ContractError, match="v1 quantity authority is quarantined"):
        m5.plan(path, tmp_path / "out")


def test_resigned_v1_content_cannot_be_admitted_as_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _plan_request(tmp_path)
    isolated_registry = tmp_path / "isolated-trust-registry.json"
    monkeypatch.setattr(m5, "TRUST_ANCHOR_REGISTRY_PATH_V1", isolated_registry)
    monkeypatch.setattr(m5, "TRUST_ANCHOR_REGISTRY_SHA256_V1", "0" * 64)
    legacy_quantities = copy.deepcopy(
        m5._quantity_authority_model()["quantities"]
    )
    for quantity in legacy_quantities:
        if quantity["producer"]["kind"] != "PROBE_FAMILY":
            quantity["upstream_family_ids"] = []
        elif quantity["producer"]["family_id"] == "d":
            quantity["upstream_family_ids"] = ["b"]
        elif quantity["producer"]["family_id"] == "f":
            quantity["upstream_family_ids"] = ["b", "d"]
        else:
            quantity["upstream_family_ids"] = []
    _resign_fixture_authority(
        request, legacy_quantities, install_registry=False
    )
    path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(m5.M5ContractError, match="PREREGISTRATION_ANCESTRY"):
        m5.plan(path, tmp_path / "out")


def test_matching_review_and_registry_cannot_cleanse_quarantined_design_input(
    tmp_path: Path,
) -> None:
    request = _plan_request(tmp_path)
    by_id = {
        identity["identity_id"]: identity for identity in request["identities"]
    }
    design_identity = by_id["fixture-scientific-design"]
    design_path = Path(design_identity["artifact_input"]["path"])
    payload = (
        _REPOSITORY_ROOT
        / "plans"
        / "sm86_calibration_microbench_design_fable5_prompt.md"
    ).read_bytes()
    design_path.write_bytes(payload)
    authority_path = Path(
        by_id["m5-quantity-authority-v2"]["artifact_input"]["path"]
    )
    authority = json.loads(authority_path.read_bytes())
    binding = authority["provenance"]["design_input_bindings"][0]
    binding["sha256"] = immutable.sha256_bytes(payload)
    binding["size_bytes"] = len(payload)
    authority_path.write_bytes(immutable.pretty_json_bytes(authority))
    _resign_current_fixture_authority(request, install_registry=False)
    path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(m5.M5ContractError, match="exactly revives"):
        m5.plan(path, tmp_path / "out")


def test_incomplete_signed_field_coverage_is_rejected(tmp_path: Path) -> None:
    request = _plan_request(tmp_path)
    authority_identity = next(
        identity
        for identity in request["identities"]
        if identity["identity_id"] == "m5-quantity-authority-v2"
    )
    authority_path = Path(authority_identity["artifact_input"]["path"])
    authority = json.loads(authority_path.read_bytes())
    authority["provenance"]["scientific_field_sources"].pop()
    authority_path.write_bytes(immutable.pretty_json_bytes(authority))
    _resign_current_fixture_authority(request, install_registry=False)
    path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(m5.M5ContractError, match="coverage is incomplete"):
        m5.plan(path, tmp_path / "out")


def test_caller_cannot_supply_a_forged_pass_audit(tmp_path: Path) -> None:
    request = _plan_request(tmp_path)
    request["identities"].append(
        _identity(
            tmp_path,
            m5.AUTHORITY_ANCESTRY_AUDIT_ID_V1,
            "AUDIT",
            "v1",
            immutable.pretty_json_bytes(
                {
                    "artifact_kind": "M5_AUTHORITY_ANCESTRY_AUDIT",
                    "verdict": "NO_LISTED_REVIVAL_DETECTED",
                }
            ),
            m5.AUTHORITY_ANCESTRY_AUDIT_MEDIA_TYPE_V1,
        )
    )
    path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(m5.M5ContractError, match="caller-supplied ancestry audit"):
        m5.plan(path, tmp_path / "out")


def test_v2_authority_requires_independent_review_witness(tmp_path: Path) -> None:
    request = _plan_request(tmp_path)
    request["identities"] = [
        identity
        for identity in request["identities"]
        if identity["identity_id"] != "m5-quantity-authority-review"
    ]
    path = _write_json(tmp_path / "request.json", request)
    with pytest.raises(m5.M5ContractError, match="INDEPENDENT_REVIEW_MISSING"):
        m5.plan(path, tmp_path / "out")


@pytest.mark.parametrize(
    ("identity_id", "field"),
    [
        ("m5-quantity-authority-review", "reviewer_id"),
        ("m5-quantity-authority-operator-approval", "operator_id"),
    ],
)
def test_self_issued_authority_witness_is_not_admitted(
    tmp_path: Path,
    identity_id: str,
    field: str,
) -> None:
    request = _plan_request(tmp_path)
    baseline_path = _write_json(tmp_path / "baseline-request.json", request)
    m5.plan(baseline_path, tmp_path / "baseline-out")
    identity = next(
        row for row in request["identities"] if row["identity_id"] == identity_id
    )
    witness_path = Path(identity["artifact_input"]["path"])
    witness = json.loads(witness_path.read_bytes())
    witness[field] = "caller-self-issued"
    witness_path.write_bytes(immutable.pretty_json_bytes(witness))
    path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(m5.M5ContractError, match="TRUST_ANCHOR_UNADMITTED"):
        m5.plan(path, tmp_path / "out")


def test_collect_cli_refuses_before_optional_imports(tmp_path: Path) -> None:
    script = Path(m5.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, "-S", str(script), "collect", "--output", str(tmp_path / "never")],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stderr == "M5_COLLECT_DISABLED\n"
    assert not (tmp_path / "never").exists()


def test_collect_function_refuses_without_side_effect(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    with pytest.raises(m5.M5ContractError, match="M5_COLLECT_DISABLED"):
        m5.collect(marker)
    assert not marker.exists()


def test_schemas_are_closed_draft_2020_12() -> None:
    for kind, path in m5.SCHEMA_PATHS.items():
        schema = json.loads(path.read_text())
        Draft202012Validator.check_schema(schema)
        assert schema["$id"] == m5.SCHEMA_IDS[kind]
        assert schema["additionalProperties"] is False
        assert immutable.sha256_file(path)[0] == m5.SCHEMA_SHA256_V1[kind]


def test_v2_authority_and_empty_public_trust_registry_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for path, schema_id, expected_sha256, controller_sha256 in (
        (
            m5.SCHEMA_PATHS_BY_VERSION[2]["M5_RUN"],
            "https://gpu-simulator.invalid/schemas/m5_calibration_run.v2.schema.json",
            "2890cc24ff7a30a63ce0ae84d0a893e1a14817272bfe1dcd58af7dd1306f710d",
            m5.SCHEMA_SHA256_BY_VERSION[2]["M5_RUN"],
        ),
        (
            m5.SCHEMA_PATHS_BY_VERSION[2]["M5_RESULT"],
            "https://gpu-simulator.invalid/schemas/m5_calibration_result.v2.schema.json",
            "a644b16b72c61aec643066e7a1b306a76f6a234a4e31be8d32847fc81fe23864",
            m5.SCHEMA_SHA256_BY_VERSION[2]["M5_RESULT"],
        ),
        (
            m5.SCHEMA_PATHS_BY_VERSION[2]["M5_DONE"],
            "https://gpu-simulator.invalid/schemas/m5_calibration_done.v2.schema.json",
            "cdf4735e5bbf14ce3f4e87932bb937074d6becb4a72ccaa30a105f2acb7edb92",
            m5.SCHEMA_SHA256_BY_VERSION[2]["M5_DONE"],
        ),
        (
            m5.AUTHORITY_SCHEMA_PATH_V2,
            "https://gpu-simulator.invalid/schemas/m5_quantity_authority.v2.schema.json",
            "ffb1a24dc2ac8bb69acf5a4d80a13b23d1f9fa1b35335638ba0e8a0bc99191e3",
            m5.AUTHORITY_SCHEMA_SHA256_V2,
        ),
        (
            m5.AUTHORITY_ANCESTRY_AUDIT_SCHEMA_PATH_V1,
            "https://gpu-simulator.invalid/schemas/m5_authority_ancestry_audit.v1.schema.json",
            "8069cabedccdce0faa55b29859ce9db4897e4b52b13a9dbfaa700643bf302fb9",
            m5.AUTHORITY_ANCESTRY_AUDIT_SCHEMA_SHA256_V1,
        ),
        (
            m5.M5_SOURCE_REGISTRY_SCHEMA_PATH_V1,
            "https://gpu-simulator.invalid/schemas/m5_source_surface_registry.v1.schema.json",
            "8400dde1a6e0a494f4f4530a5d92be871a93b291af2644a676188b74fd794b6c",
            m5.M5_SOURCE_REGISTRY_SCHEMA_SHA256_V1,
        ),
        (
            m5.TRUST_ANCHOR_SCHEMA_PATH_V1,
            "https://gpu-simulator.invalid/schemas/m5_trust_anchor_registry.v1.schema.json",
            "064fced4eac9b314ad417b63cd0edebb5aa4e5a9672288b5cccf9d7753c3dcf1",
            m5.TRUST_ANCHOR_SCHEMA_SHA256_V1,
        ),
    ):
        schema = json.loads(path.read_bytes())
        Draft202012Validator.check_schema(schema)
        assert schema["$id"] == schema_id
        assert schema["additionalProperties"] is False
        assert immutable.sha256_file(path)[0] == expected_sha256
        assert controller_sha256 == expected_sha256
    assert immutable.sha256_file(m5.M5_SOURCE_REGISTRY_PATH_V1)[0] == (
        "de937a90c920ea9eef0229bf67a0aa9348b97657f2be6643df856f945518e060"
    )
    assert m5.M5_SOURCE_REGISTRY_SHA256_V1 == (
        "de937a90c920ea9eef0229bf67a0aa9348b97657f2be6643df856f945518e060"
    )
    assert immutable.sha256_file(m5.AUTHORITY_ANCESTRY_AUDIT_TOOL_PATH_V1)[0] == (
        "99a0cdf1ea1cc777d64b87538d15a6289a64d001341b0af64d5ff5ad3406963e"
    )
    assert m5.AUTHORITY_ANCESTRY_AUDIT_TOOL_SHA256_V1 == (
        "99a0cdf1ea1cc777d64b87538d15a6289a64d001341b0af64d5ff5ad3406963e"
    )
    assert immutable.sha256_file(m5.CONTROL_JOIN_TOOL_PATH)[0] == (
        "4ea58f3928d13acf2b5cc94071d51b93935a9136118a01fe691db9652daf3bb7"
    )
    assert m5.PINNED_AUTHORITY_SHA256["control_join_tool"] == (
        "4ea58f3928d13acf2b5cc94071d51b93935a9136118a01fe691db9652daf3bb7"
    )

    assert _PRODUCTION_TRUST_REGISTRY_PATH == (
        _REPOSITORY_ROOT / "tools/local_labels/m5_trust_anchors.json"
    )
    assert immutable.sha256_file(_PRODUCTION_TRUST_REGISTRY_PATH)[0] == (
        _PRODUCTION_TRUST_REGISTRY_SHA256
    )
    monkeypatch.setattr(
        m5, "TRUST_ANCHOR_REGISTRY_PATH_V1", _PRODUCTION_TRUST_REGISTRY_PATH
    )
    monkeypatch.setattr(
        m5,
        "TRUST_ANCHOR_REGISTRY_SHA256_V1",
        _PRODUCTION_TRUST_REGISTRY_SHA256,
    )
    assert m5._trust_anchor_registry()["entries"] == []


def test_new_nested_v2_contracts_reject_unknown_properties(
    campaign: Mapping[str, Any], tmp_path: Path
) -> None:
    request = _plan_request(tmp_path / "nested-contracts")
    authority_identity = next(
        identity
        for identity in request["identities"]
        if identity["identity_id"] == m5.QUANTITY_AUTHORITY_ID_V2
    )
    authority = json.loads(
        Path(authority_identity["artifact_input"]["path"]).read_bytes()
    )
    trust_registry = m5._trust_anchor_registry()
    source_registry = json.loads(m5.M5_SOURCE_REGISTRY_PATH_V1.read_bytes())
    run = copy.deepcopy(campaign["run"])
    _published_run, _published_result, done = _bundle_models(campaign)
    cases = (
        (
            json.loads(m5.AUTHORITY_SCHEMA_PATH_V2.read_bytes()),
            authority,
            lambda model: model["provenance"].update(unexpected=True),
            "authority provenance",
        ),
        (
            json.loads(m5.AUTHORITY_SCHEMA_PATH_V2.read_bytes()),
            authority,
            lambda model: model["provenance"]["design_input_bindings"][0].update(
                unexpected=True
            ),
            "design-input binding",
        ),
        (
            json.loads(m5.AUTHORITY_SCHEMA_PATH_V2.read_bytes()),
            authority,
            lambda model: model["provenance"]["scientific_field_sources"][0].update(
                unexpected=True
            ),
            "scientific-field source",
        ),
        (
            json.loads(m5.TRUST_ANCHOR_SCHEMA_PATH_V1.read_bytes()),
            trust_registry,
            lambda model: model["entries"][0].update(unexpected=True),
            "trust entry",
        ),
        (
            json.loads(m5.TRUST_ANCHOR_SCHEMA_PATH_V1.read_bytes()),
            trust_registry,
            lambda model: model["entries"][0]["design_inputs"][0].update(
                unexpected=True
            ),
            "trust design input",
        ),
        (
            json.loads(m5.SCHEMA_PATHS_BY_VERSION[2]["M5_RUN"].read_bytes()),
            run,
            lambda model: model["provenance"].update(unexpected=True),
            "RUN provenance",
        ),
        (
            json.loads(m5.SCHEMA_PATHS_BY_VERSION[2]["M5_DONE"].read_bytes()),
            done,
            lambda model: model["artifact_inventory"][0].update(unexpected=True),
            "DONE artifact inventory",
        ),
        (
            json.loads(m5.M5_SOURCE_REGISTRY_SCHEMA_PATH_V1.read_bytes()),
            source_registry,
            lambda model: model["entries"][0].update(unexpected=True),
            "source-registry entry",
        ),
    )
    for schema, model, mutate, label in cases:
        candidate = copy.deepcopy(model)
        mutate(candidate)
        assert list(Draft202012Validator(schema).iter_errors(candidate)), label


def test_read_regular_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    original = _write(tmp_path / "original", b"bytes")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(original)
    hardlink = tmp_path / "hardlink"
    os.link(original, hardlink)
    with pytest.raises(immutable.ImmutableArtifactError, match="regular file"):
        immutable.read_regular(symlink)
    with pytest.raises(immutable.ImmutableArtifactError, match="hard-linked"):
        immutable.read_regular(hardlink)


def test_publication_is_readonly_and_idempotent(tmp_path: Path) -> None:
    payload = b"immutable\n"
    file = immutable.ImmutableFile.from_bytes("nested/value.bin", payload)
    output = tmp_path / "out"
    first = immutable.publish_directory(output, "final", [file])
    second = immutable.publish_directory(output, "final", [file])
    assert first == second
    assert (first / "nested" / "value.bin").read_bytes() == payload
    assert not (first.stat().st_mode & 0o222)
    assert not ((first / "nested" / "value.bin").stat().st_mode & 0o222)


def test_verify_pins_nested_inode_through_final_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "tree"
    nested = root / "nested"
    nested.mkdir(parents=True)
    target = _write(nested / "value", b"old")
    target.chmod(0o444)
    file = immutable.ImmutableFile("nested/value", immutable.sha256_bytes(b"old"), 3, source_path=target)
    original = immutable._verify_open_file
    replaced = False

    def replace_after_hash(descriptor: int, expected: immutable.ImmutableFile) -> bool:
        nonlocal replaced
        result = original(descriptor, expected)
        if not replaced:
            replaced = True
            target.unlink()
            target.write_bytes(b"old")
        return result

    monkeypatch.setattr(immutable, "_verify_open_file", replace_after_hash)
    with pytest.raises(immutable.ImmutableArtifactError, match="path changed"):
        immutable.verify_published_files(root, [file], require_readonly_directories=False)




def test_publication_namespace_preflight_is_aggregate_and_nonmutating(
    tmp_path: Path,
) -> None:
    output = tmp_path / "namespace"
    material = [
        immutable.ImmutableFile.from_bytes("a", b"a"),
        immutable.ImmutableFile.from_bytes("a/b", b"b"),
    ]
    with pytest.raises(immutable.ImmutableArtifactError, match="NAMESPACE_PREFIX_CONFLICT"):
        immutable.publish_directory(output, "final", material)
    assert not output.exists()

    for relative in (".hidden", "value.m5-partial", "control\x01byte"):
        candidate = tmp_path / relative.replace("/", "_")
        file = immutable.ImmutableFile.from_bytes(relative, b"x")
        with pytest.raises(immutable.ImmutableArtifactError):
            immutable.publish_directory(candidate, "final", [file])
        assert not candidate.exists()


def test_publication_name_and_leaf_byte_boundaries(tmp_path: Path) -> None:
    final_root = tmp_path / "final-boundary"
    immutable.publish_directory(
        final_root, "f" * 246,
        [immutable.ImmutableFile.from_bytes("value", b"x")],
    )
    with pytest.raises(immutable.ImmutableArtifactError, match="NAMESPACE_RESERVED_COMPONENT"):
        immutable.publish_directory(
            tmp_path / "final-overflow", "f" * 247,
            [immutable.ImmutableFile.from_bytes("value", b"x")],
        )

    leaf_root = tmp_path / "leaf-boundary"
    immutable.publish_directory(
        leaf_root, "final",
        [immutable.ImmutableFile.from_bytes("l" * 243, b"x")],
    )
    with pytest.raises(immutable.ImmutableArtifactError, match="NAMESPACE_NAME_OVERFLOW"):
        immutable.publish_directory(
            tmp_path / "leaf-overflow", "final",
            [immutable.ImmutableFile.from_bytes("l" * 244, b"x")],
        )


def test_regular_publication_and_authorization_closure_counts(
    tmp_path: Path,
) -> None:
    immutable.publish_regular_file(
        tmp_path, "value.json", b"value", partial_suffix=".m6-partial",
    )
    assert (tmp_path / "value.json").read_bytes() == b"value"

    calls: list[int] = []

    def authorize(_root: Path, descriptor: int) -> None:
        calls.append(descriptor)
        os.fstat(descriptor)

    output = tmp_path / "trees"
    file = immutable.ImmutableFile.from_bytes("value", b"value")
    immutable.publish_directory(output, "final", [file], before_commit=authorize)
    assert len(calls) == 3
    calls.clear()
    immutable.publish_directory(output, "final", [file], before_commit=authorize)
    assert len(calls) == 2


def test_verify_published_files_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = immutable.ImmutableFile.from_bytes("value", b"value")
    root = immutable.publish_directory(tmp_path / "root", "final", [file])

    def forbidden(_descriptor: int) -> None:
        raise AssertionError("verification performed fsync")

    monkeypatch.setattr(immutable.os, "fsync", forbidden)
    immutable.verify_published_files(root, [file])


@pytest.mark.parametrize("mutation", ["chmod", "replace", "mutate_restore"])
def test_public_verifier_rejects_same_uid_mutation_after_initial_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    material = (
        immutable.ImmutableFile.from_bytes("a", b"a"),
        immutable.ImmutableFile.from_bytes("z", b"z"),
    )
    root = immutable.publish_directory(tmp_path / "output", "final", material)
    early = root / "a"
    writer: int | None = None
    if mutation == "mutate_restore":
        early.chmod(0o644)
        writer = os.open(early, os.O_WRONLY)
        early.chmod(0o444)
    baseline = early.stat(follow_symlinks=False)
    original_verify = immutable._verify_open_file
    mutated = False

    def mutate_after_later_file_hash(
        descriptor: int,
        file: immutable.ImmutableFile,
    ) -> bool:
        nonlocal mutated
        result = original_verify(descriptor, file)
        if file.relative_path == "z" and not mutated:
            if mutation == "chmod":
                early.chmod(0o644)
            elif mutation == "replace":
                root.chmod(0o755)
                early.unlink()
                early.write_bytes(b"a")
                early.chmod(0o444)
                root.chmod(0o555)
            else:
                assert writer is not None
                os.pwrite(writer, b"X", 0)
                os.pwrite(writer, b"a", 0)
                os.utime(
                    early,
                    ns=(baseline.st_atime_ns, baseline.st_mtime_ns),
                )
            mutated = True
        return result

    monkeypatch.setattr(
        immutable,
        "_verify_open_file",
        mutate_after_later_file_hash,
    )
    try:
        with pytest.raises(
            immutable.ImmutableArtifactError,
            match="immutable file changed after hashing: a",
        ):
            immutable.verify_published_files(root, material)
        current = early.stat(follow_symlinks=False)
        assert mutated is True
        assert early.read_bytes() == b"a"
        if mutation == "chmod":
            assert current.st_mode & 0o222
        elif mutation == "replace":
            assert current.st_ino != baseline.st_ino
        else:
            assert current.st_ino == baseline.st_ino
            assert current.st_mtime_ns == baseline.st_mtime_ns
            assert current.st_ctime_ns != baseline.st_ctime_ns
    finally:
        if writer is not None:
            os.close(writer)
        root.chmod(0o755)


def test_publication_rejects_mid_install_partial_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    alias = tmp_path / "outside-hardlink"
    file = immutable.ImmutableFile.from_bytes("value", b"possibly unique evidence")
    original_verify = immutable._verify_locked_partial_path
    expected_partial_name = immutable._partial_name("value")
    injected = False

    def hardlink_before_reverify(
        parent_descriptor: int,
        partial_name: str,
        descriptor: int,
    ) -> None:
        nonlocal injected
        if not injected and partial_name == expected_partial_name:
            os.link(output / ".final.staging" / partial_name, alias)
            injected = True
        original_verify(parent_descriptor, partial_name, descriptor)

    monkeypatch.setattr(
        immutable,
        "_verify_locked_partial_path",
        hardlink_before_reverify,
    )
    with pytest.raises(
        immutable.ImmutableArtifactError,
        match="locked deterministic partial path changed",
    ):
        immutable.publish_directory(output, "final", [file])

    partial = output / ".final.staging" / expected_partial_name
    assert injected is True
    assert not (output / "final").exists()
    assert partial.read_bytes() == b"possibly unique evidence"
    assert alias.read_bytes() == b"possibly unique evidence"
    assert partial.stat().st_ino == alias.stat().st_ino
    assert partial.stat().st_nlink == 2


@pytest.mark.parametrize("lock_kind", ["stage", "partial"])
def test_publication_rejects_live_lock_contention(
    tmp_path: Path,
    lock_kind: str,
) -> None:
    output = tmp_path / "output"
    stage = output / ".final.staging"
    stage.mkdir(parents=True)
    file = immutable.ImmutableFile.from_bytes("value", b"possibly unique evidence")
    sentinel: Path | None = None
    if lock_kind == "stage":
        sentinel = stage / "sentinel"
        sentinel.write_bytes(b"retained stage evidence")
        sentinel.chmod(0o444)
        locked_path = stage
        expected = (
            "immutable publication is owned by a live publisher: .final.staging"
        )
        flags = os.O_RDONLY | os.O_DIRECTORY
    else:
        locked_path = stage / immutable._partial_name("value")
        locked_path.write_bytes(b"possibly unique evidence")
        locked_path.chmod(0o444)
        expected = (
            "deterministic partial is owned by a live publisher: "
            f"{locked_path.name}"
        )
        flags = os.O_RDONLY

    descriptor = os.open(
        locked_path,
        flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    locked_before = immutable._snapshot_fields(os.fstat(descriptor))
    inventory_before = tuple(sorted(path.name for path in stage.iterdir()))
    stage_before = immutable._snapshot_fields(stage.stat(follow_symlinks=False))
    sentinel_before = (
        immutable._snapshot_fields(sentinel.stat(follow_symlinks=False))
        if sentinel is not None
        else None
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(immutable.ImmutableArtifactError) as error:
            immutable.publish_directory(output, "final", [file])
        assert immutable._snapshot_fields(os.fstat(descriptor)) == locked_before
        assert immutable._snapshot_fields(
            locked_path.stat(follow_symlinks=False)
        ) == locked_before
        assert immutable._snapshot_fields(
            stage.stat(follow_symlinks=False)
        ) == stage_before
        assert tuple(sorted(path.name for path in stage.iterdir())) == inventory_before
        if sentinel is not None:
            assert sentinel_before is not None
            assert immutable._snapshot_fields(
                sentinel.stat(follow_symlinks=False)
            ) == sentinel_before
            assert sentinel.read_bytes() == b"retained stage evidence"
        else:
            assert locked_before[3] == 1
            assert locked_path.read_bytes() == b"possibly unique evidence"
    finally:
        os.close(descriptor)

    assert str(error.value) == expected
    assert not (output / "final").exists()


def test_equal_concurrent_publisher_accepts_durable_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    file = immutable.ImmutableFile.from_bytes("value", b"possibly unique evidence")
    committed = threading.Event()
    release = threading.Event()
    first_results: list[Path] = []
    first_errors: list[BaseException] = []
    durability_calls: list[tuple[int, int]] = []
    authorization_calls = 0
    original_durability_proof = immutable._durability_proof_at

    def pause_after_commit(_root: Path, _descriptor: int) -> None:
        nonlocal authorization_calls
        authorization_calls += 1
        if authorization_calls == 3:
            committed.set()
            if not release.wait(timeout=10):
                raise AssertionError("concurrent-publisher release timed out")

    def record_durability_proof(
        publication_root_descriptor: int,
        tree_root_descriptor: int,
        material: tuple[immutable.ImmutableFile, ...],
    ):
        durability_calls.append(
            (publication_root_descriptor, tree_root_descriptor)
        )
        return original_durability_proof(
            publication_root_descriptor,
            tree_root_descriptor,
            material,
        )

    def publish_first() -> None:
        try:
            first_results.append(
                immutable.publish_directory(
                    output,
                    "final",
                    [file],
                    before_commit=pause_after_commit,
                )
            )
        except BaseException as exc:
            first_errors.append(exc)

    monkeypatch.setattr(
        immutable,
        "_durability_proof_at",
        record_durability_proof,
    )
    publisher = threading.Thread(target=publish_first)
    publisher.start()
    try:
        assert committed.wait(timeout=10), "first publisher did not commit"
        second_result = immutable.publish_directory(output, "final", [file])
    finally:
        release.set()
        publisher.join(timeout=10)

    assert not publisher.is_alive()
    assert first_errors == []
    assert first_results == [output / "final"]
    assert second_result == output / "final"
    assert authorization_calls == 3
    assert len(durability_calls) == 1
    assert durability_calls[0][0] != durability_calls[0][1]
    assert not (output / ".final.staging").exists()
    immutable.verify_published_files(output / "final", [file])


@pytest.mark.parametrize("conflict_kind", ["content", "inventory"])
def test_stale_stage_conflicts_preserve_first_publisher_evidence(
    tmp_path: Path,
    conflict_kind: str,
) -> None:
    output = tmp_path / "output"
    first_payload = b"possibly unique evidence"
    first_files = [immutable.ImmutableFile.from_bytes("value", first_payload)]
    if conflict_kind == "content":
        second_payload = bytes([first_payload[0] ^ 1]) + first_payload[1:]
        second_files = [
            immutable.ImmutableFile.from_bytes("value", second_payload)
        ]
        expected_error = "staged file collision: value"
    else:
        first_files.append(
            immutable.ImmutableFile.from_bytes("extra", b"retained extra evidence")
        )
        second_files = first_files[:1]
        expected_error = "deterministic stage inventory mismatch"
    authorization_calls = 0

    def stop_before_commit(_root: Path, _descriptor: int) -> None:
        nonlocal authorization_calls
        authorization_calls += 1
        if authorization_calls == 2:
            raise RuntimeError("injected stop before commit")

    with pytest.raises(RuntimeError, match="injected stop before commit"):
        immutable.publish_directory(
            output,
            "final",
            first_files,
            before_commit=stop_before_commit,
        )
    assert authorization_calls == 2
    stage = output / ".final.staging"
    inventory_before = tuple(sorted(path.name for path in stage.iterdir()))
    stage_before = immutable._snapshot_fields(stage.stat(follow_symlinks=False))
    files_before = {
        path.name: (
            immutable._snapshot_fields(path.stat(follow_symlinks=False)),
            path.read_bytes(),
        )
        for path in stage.iterdir()
    }

    with pytest.raises(immutable.ImmutableArtifactError) as error:
        immutable.publish_directory(output, "final", second_files)

    assert expected_error in str(error.value)
    assert not (output / "final").exists()
    assert immutable._snapshot_fields(
        stage.stat(follow_symlinks=False)
    ) == stage_before
    assert tuple(sorted(path.name for path in stage.iterdir())) == inventory_before
    assert {
        path.name: (
            immutable._snapshot_fields(path.stat(follow_symlinks=False)),
            path.read_bytes(),
        )
        for path in stage.iterdir()
    } == files_before
    immutable.verify_published_files(stage, first_files)


def test_publication_rejects_stage_and_final_symlink_names(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (output / ".final.staging").symlink_to(target, target_is_directory=True)
    file = immutable.ImmutableFile.from_bytes("value", b"value")
    with pytest.raises(immutable.ImmutableArtifactError, match="deterministic stage"):
        immutable.publish_directory(output, "final", [file])

    final_output = tmp_path / "final-output"
    final_output.mkdir()
    (final_output / "final").symlink_to(target, target_is_directory=True)
    with pytest.raises(immutable.ImmutableArtifactError, match="concurrent final winner"):
        immutable.publish_directory(final_output, "final", [file])


def test_publication_rejects_final_and_stage_coexistence(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "final").mkdir()
    (output / ".final.staging").mkdir()
    file = immutable.ImmutableFile.from_bytes("value", b"value")
    with pytest.raises(immutable.ImmutableArtifactError, match="coexist"):
        immutable.publish_directory(output, "final", [file])


def test_relative_delta_rejects_overflow() -> None:
    with pytest.raises(m5.M5ContractError, match="overflowed"):
        m5._relative_delta(1.7e308, -1.7e308)


def test_collect_is_only_disabled_subcommand() -> None:
    help_text = m5._parser().format_help()
    assert all(command in help_text for command in ("plan", "analyze", "verify", "collect"))
