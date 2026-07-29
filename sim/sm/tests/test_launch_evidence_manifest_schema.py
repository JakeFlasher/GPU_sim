from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema


SCHEMA_PATH = Path(__file__).parents[1] / "schema" / "launch_evidence_manifest.v1.schema.json"
SHA = "0" * 64


def artifact(path: str) -> dict[str, object]:
    return {"path": path, "size_bytes": 1, "sha256": SHA}


def manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifacts": {
            "dynamic_trace_pb": artifact("trace/dynamic_trace.pb"),
            "threadblock_payload": {
                "path": "trace/threadblocks",
                "file_count": 1,
                "raw_bytes": 1,
                "sorted_path_manifest_sha256": SHA,
                "sorted_path_stream_sha256": SHA,
            },
            "allocations_json": {
                "status": "present",
                "artifact": artifact("trace/extra_info/allocations.json"),
            },
            "trace_sampling_json": {"status": "absent"},
            "enhanced_execution_info_json": artifact(
                "trace/extra_info/enhanced_execution_info.json"
            ),
            "control_bits_sidecar": artifact("control/bench.control_bits.json"),
            "arch_card": artifact("cards/sm_86.json"),
            "pchase_card": artifact("cards/sm_86_static_card.json"),
        },
        "launch_selector": {
            "device_id": 0,
            "stream_id": 0,
            "kernel_id": 1,
            "function_unique_id": 1,
        },
        "architecture": {"arch_id": "sm_86", "sku": "rtx_3090_ti"},
        "binary_digest": {
            "status": "typed_gap",
            "gap": "trace_contract_does_not_emit_binary_digest",
        },
    }


def validator() -> jsonschema.Draft7Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft7Validator.check_schema(schema)
    return jsonschema.Draft7Validator(schema)


def test_launch_evidence_schema_admits_exact_v1_contract() -> None:
    instance = manifest()
    validator().validate(instance)

    value_arm = copy.deepcopy(instance)
    value_arm["binary_digest"] = {"status": "value", "sha256": SHA}
    validator().validate(value_arm)


def test_launch_evidence_schema_rejects_contaminated_shapes() -> None:
    base = manifest()
    mutations = []

    unknown = copy.deepcopy(base)
    unknown["launch_selector"]["unexpected"] = True
    mutations.append(unknown)

    uppercase = copy.deepcopy(base)
    uppercase["artifacts"]["arch_card"]["sha256"] = "A" * 64
    mutations.append(uppercase)

    traversal = copy.deepcopy(base)
    traversal["artifacts"]["arch_card"]["path"] = "cards/../cards/sm_86.json"
    mutations.append(traversal)

    interior_dot = copy.deepcopy(base)
    interior_dot["artifacts"]["arch_card"]["path"] = "cards/./sm_86.json"
    mutations.append(interior_dot)

    terminal_newline = copy.deepcopy(base)
    terminal_newline["artifacts"]["arch_card"]["path"] = "cards/sm_86.json\n"
    mutations.append(terminal_newline)

    alias_path = copy.deepcopy(base)
    alias_path["artifacts"]["arch_card"]["path"] = "cards//sm_86.json"
    mutations.append(alias_path)

    absent_with_artifact = copy.deepcopy(base)
    absent_with_artifact["artifacts"]["trace_sampling_json"]["artifact"] = artifact(
        "trace/trace_sampling.json"
    )
    mutations.append(absent_with_artifact)

    cross_arch = copy.deepcopy(base)
    cross_arch["architecture"]["sku"] = "rtx_5060_laptop"
    mutations.append(cross_arch)

    empty_payload = copy.deepcopy(base)
    empty_payload["artifacts"]["threadblock_payload"]["file_count"] = 0
    mutations.append(empty_payload)

    out_of_range_device = copy.deepcopy(base)
    out_of_range_device["launch_selector"]["device_id"] = 2**63
    mutations.append(out_of_range_device)

    out_of_range_size = copy.deepcopy(base)
    out_of_range_size["artifacts"]["dynamic_trace_pb"]["size_bytes"] = 2**64
    mutations.append(out_of_range_size)

    for contaminated in mutations:
        assert list(validator().iter_errors(contaminated)), contaminated
