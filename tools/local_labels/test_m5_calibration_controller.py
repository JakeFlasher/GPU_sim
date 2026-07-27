from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
from jsonschema import Draft202012Validator

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import m5_calibration_controller as m5


_COMMIT = "1" * 40
_SEED = bytes(range(32))


def _encode_point(point: tuple[int, int, int, int]) -> bytes:
    x, y, z, _t = point
    inverse = pow(z, m5._ED25519_Q - 2, m5._ED25519_Q)
    x = x * inverse % m5._ED25519_Q
    y = y * inverse % m5._ED25519_Q
    encoded = bytearray(y.to_bytes(32, "little"))
    encoded[31] |= (x & 1) << 7
    return bytes(encoded)


def _test_key() -> tuple[bytes, int, bytes]:
    digest = hashlib.sha512(_SEED).digest()
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    public = _encode_point(m5._ed25519_scalar_mult(m5._ED25519_BASE, scalar))
    return public, scalar, digest[32:]


def _sign(message: bytes) -> bytes:
    public, scalar, prefix = _test_key()
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


def _plan_request(root: Path) -> dict[str, Any]:
    public, _scalar, _prefix = _test_key()
    quantities = copy.deepcopy(m5._quantity_authority_model()["quantities"])
    authority_unsigned = {
        "schema_version": 2,
        "artifact_kind": "M5_QUANTITY_AUTHORITY",
        "authority_id": "sm86-m5-quantity-authority-v2",
        "execution_order": list(m5.EXECUTION_ORDER),
        "scientific_rank": list(m5.SCIENTIFIC_RANK),
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
    rank = {family: index + 1 for index, family in enumerate(m5.SCIENTIFIC_RANK)}
    families = [
        {
            "family_id": family,
            "execution_index": index,
            "scientific_rank": rank[family],
            "harness_identity_ids": ["harness-main"],
            "build_relationship_ids": ["build-rel-main"],
        }
        for index, family in enumerate(m5.EXECUTION_ORDER, start=1)
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
        "family_execution_order": list(m5.EXECUTION_ORDER),
        "family_scientific_rank": list(m5.SCIENTIFIC_RANK),
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
                if producer_family == "d":
                    upstream_decision_sha256s = sorted(decision_hashes_by_family.get("b", []))
                elif producer_family == "f":
                    upstream_decision_sha256s = sorted(
                        decision_hashes_by_family.get("b", [])
                        + decision_hashes_by_family.get("d", [])
                    )
                else:
                    upstream_decision_sha256s = []
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
def _stable_source() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(m5, "_observe_git_source", lambda: (_COMMIT, b""))
    yield
    monkeypatch.undo()


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


def test_plan_analyze_relocate_and_verify(campaign: Mapping[str, Any], tmp_path: Path) -> None:
    run, result, done = _bundle_models(campaign)
    assert run["family_execution_order"] == ["b", "a", "d", "f", "c", "e"]
    assert run["family_scientific_rank"] == ["b", "f", "a", "d", "c", "e"]
    assert run["counts"]["quantity_count"] == 31
    assert run["counts"]["canonical_runnable_cell_count"] == 122
    assert result["counts"]["measured_quantity_count"] == 31
    assert result["counts"]["canonical_control_sidecar_count"] == 122
    assert result["components"]["pchase"]["property_count"] == 14
    assert done["counts"]["authority_receipt_count"] == len(result["authenticity"]["entries"])
    relocated = tmp_path / "relocated"
    shutil.copytree(campaign["bundle"], relocated)
    verified = m5.verify(relocated, campaign["authority_sha256"])
    assert verified["model_seal_sha256"] == done["model_seal_sha256"]


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
    assert protocol["repetition_count"] == expected == 186
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
