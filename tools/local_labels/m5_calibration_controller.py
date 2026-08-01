#!/usr/bin/env python3
"""Hardware-free M5 calibration planning, analysis, and verification controller."""

from __future__ import annotations

# Direct-script collection refusal is intentionally before optional/heavy imports,
# repository path resolution, parser construction, or any filesystem operation.
if __name__ == "__main__":
    import sys as _bootstrap_sys

    if len(_bootstrap_sys.argv) > 1 and _bootstrap_sys.argv[1] == "collect":
        _bootstrap_sys.stderr.write("M5_COLLECT_DISABLED\n")
        raise SystemExit(2)

import argparse
import copy
import functools
import hashlib
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import jsonschema
from jsonschema import validators

try:
    from tools.local_labels import immutable_artifact as immutable
except ModuleNotFoundError:  # Direct execution from tools/local_labels.
    import immutable_artifact as immutable  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
SCHEMA_PATHS_BY_VERSION = {
    1: {
        "M5_RUN": SCHEMA_ROOT / "m5_calibration_run.v1.schema.json",
        "M5_RESULT": SCHEMA_ROOT / "m5_calibration_result.v1.schema.json",
        "M5_DONE": SCHEMA_ROOT / "m5_calibration_done.v1.schema.json",
    },
    2: {
        "M5_RUN": SCHEMA_ROOT / "m5_calibration_run.v2.schema.json",
        "M5_RESULT": SCHEMA_ROOT / "m5_calibration_result.v2.schema.json",
        "M5_DONE": SCHEMA_ROOT / "m5_calibration_done.v2.schema.json",
    },
}
SCHEMA_IDS_BY_VERSION = {
    version: {
        kind: f"https://gpu-simulator.invalid/schemas/m5_calibration_{kind.removeprefix('M5_').lower()}.v{version}.schema.json"
        for kind in ("M5_RUN", "M5_RESULT", "M5_DONE")
    }
    for version in (1, 2)
}
# These are immutable compatibility pins.  v1 is retained for historical
# bundles; new planning and analysis use only the explicit v2 dispatch.
SCHEMA_SHA256_BY_VERSION = {
    1: {
        "M5_RUN": "e369346d71d108bdef1f49bd9beae195fbb30553d4c383b92d5486796ef7038c",
        "M5_RESULT": "054adda2fdd3d1b4250837934e81a32869968ae06383c21766e24da3cff1b53b",
        "M5_DONE": "96d25e640abe0278a6aee901bf55548f68bfd7f4ad85137729dff39f1d1ea029",
    },
    2: {
        "M5_RUN": "2890cc24ff7a30a63ce0ae84d0a893e1a14817272bfe1dcd58af7dd1306f710d",
        "M5_RESULT": "a644b16b72c61aec643066e7a1b306a76f6a234a4e31be8d32847fc81fe23864",
        "M5_DONE": "cdf4735e5bbf14ce3f4e87932bb937074d6becb4a72ccaa30a105f2acb7edb92",
    },
}
# Compatibility aliases are deliberately v1-only and are not used by active
# production dispatch.
SCHEMA_PATHS = SCHEMA_PATHS_BY_VERSION[1]
SCHEMA_IDS = SCHEMA_IDS_BY_VERSION[1]
SCHEMA_SHA256_V1 = SCHEMA_SHA256_BY_VERSION[1]
SCHEMA_PATHS_V1 = SCHEMA_PATHS_BY_VERSION[1]
SCHEMA_PATHS_V2 = SCHEMA_PATHS_BY_VERSION[2]
SCHEMA_IDS_V1 = SCHEMA_IDS_BY_VERSION[1]
SCHEMA_IDS_V2 = SCHEMA_IDS_BY_VERSION[2]
SCHEMA_SHA256_V2 = SCHEMA_SHA256_BY_VERSION[2]
ACTIVE_SCHEMA_VERSION = 2
QUANTITY_AUTHORITY_ID_V1 = "m5-quantity-authority"
QUANTITY_AUTHORITY_ID_V2 = "m5-quantity-authority-v2"
QUANTITY_AUTHORITY_MEDIA_TYPE_V2 = "application/vnd.m5.quantity-authority+json"
AUTHORITY_ANCESTRY_ID_V2 = "m5-quantity-authority-ancestry"
AUTHORITY_ANCESTRY_AUDIT_ID_V1 = "m5-quantity-authority-ancestry-audit"
AUTHORITY_ANCESTRY_AUDIT_MEDIA_TYPE_V1 = (
    "application/vnd.m5.authority-ancestry-audit+json"
)
AUTHORITY_REVIEW_ID_V2 = "m5-quantity-authority-review"
AUTHORITY_OPERATOR_APPROVAL_ID_V2 = "m5-quantity-authority-operator-approval"
_AUTHORITY_SIGNATURE_DOMAIN = b"M5_QUANTITY_AUTHORITY_V2\0"
AUTHORITY_SCHEMA_PATH_V2 = SCHEMA_ROOT / "m5_quantity_authority.v2.schema.json"
AUTHORITY_SCHEMA_ID_V2 = "https://gpu-simulator.invalid/schemas/m5_quantity_authority.v2.schema.json"
AUTHORITY_SCHEMA_SHA256_V2 = "ffb1a24dc2ac8bb69acf5a4d80a13b23d1f9fa1b35335638ba0e8a0bc99191e3"
AUTHORITY_ANCESTRY_AUDIT_TOOL_PATH_V1 = (
    REPO_ROOT / "tools" / "local_labels" / "m5_authority_ancestry_audit.py"
)
AUTHORITY_ANCESTRY_AUDIT_SCHEMA_PATH_V1 = (
    SCHEMA_ROOT / "m5_authority_ancestry_audit.v1.schema.json"
)
AUTHORITY_ANCESTRY_AUDIT_SCHEMA_ID_V1 = (
    "https://gpu-simulator.invalid/schemas/m5_authority_ancestry_audit.v1.schema.json"
)
AUTHORITY_ANCESTRY_AUDIT_TOOL_SHA256_V1 = "99a0cdf1ea1cc777d64b87538d15a6289a64d001341b0af64d5ff5ad3406963e"
AUTHORITY_ANCESTRY_AUDIT_SCHEMA_SHA256_V1 = "8069cabedccdce0faa55b29859ce9db4897e4b52b13a9dbfaa700643bf302fb9"
M5_SOURCE_REGISTRY_PATH_V1 = (
    REPO_ROOT / "tools" / "local_labels" / "m5_source_surface_registry.json"
)
M5_SOURCE_REGISTRY_SCHEMA_PATH_V1 = (
    SCHEMA_ROOT / "m5_source_surface_registry.v1.schema.json"
)
M5_SOURCE_REGISTRY_SCHEMA_ID_V1 = (
    "https://gpu-simulator.invalid/schemas/m5_source_surface_registry.v1.schema.json"
)
M5_SOURCE_REGISTRY_SHA256_V1 = "de937a90c920ea9eef0229bf67a0aa9348b97657f2be6643df856f945518e060"
M5_SOURCE_REGISTRY_SCHEMA_SHA256_V1 = "8400dde1a6e0a494f4f4530a5d92be871a93b291af2644a676188b74fd794b6c"
TRUST_ANCHOR_REGISTRY_PATH_V1 = REPO_ROOT / "tools" / "local_labels" / "m5_trust_anchors.json"
TRUST_ANCHOR_REGISTRY_MEDIA_TYPE_V1 = (
    "application/vnd.m5.trust-anchor-registry+json"
)
TRUST_ANCHOR_REGISTRY_SHA256_V1 = (
    "723bd12653f0aa1d4875ed731bdf8e2ffd001acc8c46ed33fd1d8d3ca893710e"
)
TRUST_ANCHOR_SCHEMA_PATH_V1 = SCHEMA_ROOT / "m5_trust_anchor_registry.v1.schema.json"
TRUST_ANCHOR_SCHEMA_ID_V1 = (
    "https://gpu-simulator.invalid/schemas/m5_trust_anchor_registry.v1.schema.json"
)
TRUST_ANCHOR_SCHEMA_SHA256_V1 = "064fced4eac9b314ad417b63cd0edebb5aa4e5a9672288b5cccf9d7753c3dcf1"
CAMPAIGN_PROTOCOL_PATH = REPO_ROOT / "tools" / "local_labels" / "campaign_protocol_pins.json"
CANONICAL_MATRIX_PATH = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "campaign"
    / "sm86_reference_capture_matrix.json"
)
CANONICAL_ROSTER_PATH = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "campaign"
    / "capture_roster_authority_sm86.json"
)
CONTROL_EMITTER_PATH = REPO_ROOT / "tools" / "sass_control" / "emit_control_bits_sidecar.py"
CONTROL_EXTRACTOR_PATH = REPO_ROOT / "tools" / "sass_control" / "extractor.py"
CONTROL_JOIN_TOOL_PATH = REPO_ROOT / "tools" / "sass_control" / "run_pc_join_witness.py"
CONTROL_JOIN_SCHEMA_PATH = (
    SCHEMA_ROOT / "sass_control_pc_join_witness.schema.json"
)
PCHASE_SOURCE_PATH = REPO_ROOT / "tools" / "pchase" / "pchase_kernels.cu"
PCHASE_V1_CARD_PATH = REPO_ROOT / "data" / "pchase" / "sm_86_static_card.json"
PINNED_AUTHORITY_SHA256 = {
    "campaign_protocol": "3ec28feb6eab9dd3943a021a96afb4cb0562d469c929f673d17b0f7c9cb9545a",
    "canonical_matrix": "af29d72e955cf42e281036190f2d55d72af0af92a632c78c8d159073a072fb0e",
    "canonical_roster": "d02c817091b0a9f06c2106c0987c59a481910afcc3b78ab1de7a7d5bd5a6b4b4",
    "control_emitter": "57b980957972e4032e533b36b88c24cf5d96bd01fddf547efb26c68bf82cf6a2",
    "control_extractor": "98df23450dc5790a1000e41739f78d212ae587c6a0eda9a0c55a33259ac0b4d2",
    "control_join_tool": "4ea58f3928d13acf2b5cc94071d51b93935a9136118a01fe691db9652daf3bb7",
    "control_join_schema": "f40c14e9fb128cab1f4ed001a204f4ce47d4c945bf18deb8f096a405dcb78359",
    "pchase_source": "ce1ecc34bdee6c3542ba6935d60e9766fcbd0aa52f5a74299090a440c5fa86cd",
    "pchase_v1_card": "e87d2e208f5326a8083d1a54799864953a62ed1fece1b628583cf8d5cafc581d",
}
CONTROL_PARSER_CONTRACT = {
    "supported_toolkit_versions": ["V13.2"],
    "supported_arches": ["sm_80", "sm_86"],
    "control_layout": {
        "source": "high_word >> 41 (23 bits)",
        "stall": "bits 0:3",
        "yield_flag": "bit 4",
        "write_barrier": "bits 5:7 (7 = none)",
        "read_barrier": "bits 8:10 (7 = none)",
        "wait_mask": "bits 11:16",
        "reuse": "bits 17:20",
        "reserved_must_be_zero": "bits 21:22",
    },
}
EXECUTION_ORDER = ("b", "a", "d", "f", "c", "e")
SCIENTIFIC_RANK = ("b", "f", "a", "d", "c", "e")
OUTCOME_KINDS = frozenset(
    {"MEASURED", "RANGE", "CENSORED", "MIXTURE", "UNRESOLVED_STRUCTURE", "SURFACE_GAP"}
)
TYPED_GAPS = OUTCOME_KINDS - {"MEASURED"}
EVIDENCE_ROLES = ("curve", "control", "confound", "composition")
M5_COLLECT_DISABLED = "M5_COLLECT_DISABLED"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_IDENTITY_KINDS = frozenset(
    {
        "TOOL",
        "HARNESS",
        "SOURCE",
        "BINARY",
        "BUILD",
        "DERIVATION",
        "SCHEMA",
        "ROSTER",
        "PROTOCOL",
        "AUTHORITY",
        "SOURCE_STATUS",
        "DESIGN_INPUT",
        "AUDIT",
    }
)
_BINDING_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "IDENTITY",
            "EVIDENCE_CURVE",
            "EVIDENCE_CONTROL",
            "EVIDENCE_CONFOUND",
            "EVIDENCE_COMPOSITION",
            "NCU_CROSS_CHECK",
            "NCU_VERDICT",
            "READOUT_PRODUCER",
            "RAW_SAMPLE_SERIES",
            "QUANTITY_DECISION",
            "PROTOCOL_STATE_WITNESS",
            "CONTROL_PC_JOIN_WITNESS",
            "CONTROL_INVENTORY",
            "CONTROL_SIDECAR",
            "PCHASE_FRESH_CARD",
            "PRE_NCU_CANDIDATE",
            "AUTHENTICATED_INPUT",
            "AUTHORITY_RECEIPT",
        )
    )
}
_AUTO_IDENTITIES = {
    "m5-controller",
    "m5-immutable-helper",
    "m5-run-schema",
    "m5-result-schema",
    "m5-done-schema",
    "m5-campaign-protocol",
    "m5-canonical-matrix",
    "m5-canonical-roster",
    QUANTITY_AUTHORITY_ID_V1,
    "m5-quantity-authority-schema",
    "m5-ancestry-audit-tool",
    "m5-ancestry-audit-schema",
    "m5-source-surface-registry",
    "m5-source-surface-registry-schema",
    "m5-control-emitter",
    "m5-control-extractor",
    "m5-control-join-tool",
    "m5-control-join-schema",
    "m5-pchase-source",
    "m5-pchase-v1-card",
    "m5-source-status",
}
_EMPTY_SHA256 = immutable.sha256_bytes(b"")
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
_RECEIPT_DOMAIN = b"M5_AUTHORITY_RECEIPT_V1\0"
_ED25519_Q = 2**255 - 19
_ED25519_L = 2**252 + 27742317777372353535851937790883648493
_ED25519_D = (-121665 * pow(121666, _ED25519_Q - 2, _ED25519_Q)) % _ED25519_Q
_ED25519_I = pow(2, (_ED25519_Q - 1) // 4, _ED25519_Q)


def _ed25519_xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_ED25519_D * y * y + 1, _ED25519_Q - 2, _ED25519_Q)
    x = pow(xx % _ED25519_Q, (_ED25519_Q + 3) // 8, _ED25519_Q)
    if (x * x - xx) % _ED25519_Q != 0:
        x = x * _ED25519_I % _ED25519_Q
    if (x * x - xx) % _ED25519_Q != 0:
        _fail("invalid Ed25519 point encoding")
    return x


def _ed25519_decode_point(encoded: bytes, label: str) -> tuple[int, int, int, int]:
    if len(encoded) != 32:
        _fail(f"{label} must be exactly 32 bytes")
    raw = int.from_bytes(encoded, "little")
    sign = raw >> 255
    y = raw & ((1 << 255) - 1)
    if y >= _ED25519_Q:
        _fail(f"{label} is not a canonical Ed25519 point")
    x = _ed25519_xrecover(y)
    if x & 1 != sign:
        x = _ED25519_Q - x
    if (y * y - x * x - 1 - _ED25519_D * x * x * y * y) % _ED25519_Q:
        _fail(f"{label} is not on the Ed25519 curve")
    return x, y, 1, x * y % _ED25519_Q


def _ed25519_add(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = (y1 - x1) * (y2 - x2) % _ED25519_Q
    b = (y1 + x1) * (y2 + x2) % _ED25519_Q
    c = 2 * _ED25519_D * t1 * t2 % _ED25519_Q
    d = 2 * z1 * z2 % _ED25519_Q
    e = (b - a) % _ED25519_Q
    f = (d - c) % _ED25519_Q
    g = (d + c) % _ED25519_Q
    h = (b + a) % _ED25519_Q
    return e * f % _ED25519_Q, g * h % _ED25519_Q, f * g % _ED25519_Q, e * h % _ED25519_Q


def _ed25519_scalar_mult(
    point: tuple[int, int, int, int], scalar: int
) -> tuple[int, int, int, int]:
    result = (0, 1, 1, 0)
    addend = point
    while scalar:
        if scalar & 1:
            result = _ed25519_add(result, addend)
        addend = _ed25519_add(addend, addend)
        scalar >>= 1
    return result


_ED25519_BASE = _ed25519_decode_point(
    bytes.fromhex("5866666666666666666666666666666666666666666666666666666666666666"),
    "Ed25519 base point",
)


def _ed25519_equal(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    return (
        (left[0] * right[2] - right[0] * left[2]) % _ED25519_Q == 0
        and (left[1] * right[2] - right[1] * left[2]) % _ED25519_Q == 0
    )


def _verify_ed25519(public_key: bytes, message: bytes, signature: bytes) -> None:
    if len(signature) != 64:
        _fail("authority receipt signature must be exactly 64 bytes")
    public_point = _ed25519_decode_point(public_key, "authority public key")
    r_point = _ed25519_decode_point(signature[:32], "authority receipt R")
    scalar = int.from_bytes(signature[32:], "little")
    if scalar >= _ED25519_L:
        _fail("authority receipt signature scalar is noncanonical")
    identity = (0, 1, 1, 0)
    if _ed25519_equal(public_point, identity) or not _ed25519_equal(
        _ed25519_scalar_mult(public_point, _ED25519_L), identity
    ):
        _fail("authority public key is not in the prime-order subgroup")
    if _ed25519_equal(r_point, identity) or not _ed25519_equal(
        _ed25519_scalar_mult(r_point, _ED25519_L), identity
    ):
        _fail("authority receipt R is not in the prime-order subgroup")
    challenge = int.from_bytes(
        hashlib.sha512(signature[:32] + public_key + message).digest(), "little"
    ) % _ED25519_L
    if not _ed25519_equal(
        _ed25519_scalar_mult(_ED25519_BASE, scalar),
        _ed25519_add(r_point, _ed25519_scalar_mult(public_point, challenge)),
    ):
        _fail("authority receipt Ed25519 signature mismatch")


def _validate_ed25519_public_key(public_key: bytes, label: str) -> None:
    point = _ed25519_decode_point(public_key, label)
    identity = (0, 1, 1, 0)
    if _ed25519_equal(point, identity) or not _ed25519_equal(
        _ed25519_scalar_mult(point, _ED25519_L), identity
    ):
        _fail(f"{label} is not a nonidentity prime-order Ed25519 point")


def _quantity_row(
    quantity_id: str,
    quantity_group: str,
    family_id: str | None,
    consumer_binding: str,
    unit: str,
    meaning: str,
    access_class: str,
    gap_kinds: tuple[str, ...],
    methods: tuple[str, str],
    *,
    agreement_limit: float = 0.10,
) -> dict[str, Any]:
    producer = (
        {"kind": "PROBE_FAMILY", "family_id": family_id}
        if family_id is not None
        else {"kind": "PCHASE", "component_id": "pchase"}
    )
    contracts = []
    role_sets = (("curve", "control"), ("confound", "composition"))
    for index, (method, roles) in enumerate(zip(methods, role_sets, strict=True), start=1):
        instrument = "THROUGHPUT" if method.startswith("THROUGHPUT") or method in {
            "BURST_SLOPE_BREAK",
            "LITTLE_LAW_TRIANGULATION",
        } else "CLOCK64"
        readout_id = f"{quantity_id}.r{index}"
        contracts.append(
            {
                "readout_id": readout_id,
                "method_class": method,
                "instrument": instrument,
                "readout_role": "VALUE",
                "derivation_identity_id": f"{readout_id}.derivation",
                # The two readouts must have different estimator forms. The
                # second contract uses a mean rather than silently repeating
                # the first median estimator.
                "derivation_operation": "MEDIAN" if index == 1 else "MEAN",
                "sample_operation": "RATIO",
                "method_contract": {
                    "mechanism": method,
                    "estimator_identity": (
                        "RATIO_THEN_MEDIAN" if index == 1 else "RATIO_THEN_MEAN"
                    ),
                    "confounder_set": list(roles),
                    "control_evidence_roles": list(roles),
                    "falsification_rule": "MAX_RELATIVE_DELTA_WITHIN_AGREEMENT_LIMIT",
                },
                "required_evidence_roles": list(roles),
            }
        )
    if unit == "probability":
        value_domain = {"kind": "PROBABILITY", "integral": False}
    elif unit == "structure":
        value_domain = {"kind": "NONNEGATIVE", "integral": False}
    else:
        integral_units = {"bytes", "sectors", "lines", "ways", "entries"}
        value_domain = {"kind": "POSITIVE", "integral": unit in integral_units}
    return {
        "quantity_id": quantity_id,
        "quantity_group": quantity_group,
        "producer": producer,
        "consumer_binding": consumer_binding,
        "meaning": meaning,
        "conditioning": {
            "residency": "NOT_APPLICABLE" if family_id is None else "MIXED",
            "access_class": access_class,
            "scope": "sm_86",
            "variant": quantity_id,
        },
        "unit": unit,
        "value_domain": value_domain,
        "agreement_limit_fraction": agreement_limit,
        "readout_contracts": contracts,
        "authorized_gap_kinds": list(gap_kinds),
        "required": True,
    }


def _quantity_authority_model() -> dict[str, Any]:
    clock_rtt = ("CLOCK64_DEPENDENCY_CHAIN", "RTT_CONTRAST")
    clock_control = ("CLOCK64_ELAPSED_WINDOW", "PERSISTENCE_DELTA_CONTRAST")
    throughput = ("THROUGHPUT_PLATEAU", "BURST_SLOPE_BREAK")
    rows = [
        _quantity_row("b.l1_xbar_read_port_width", "b1", "b", "sim.l1_xbar.read_port_width", "B/SM-cycle", "RTT-invariant per-SM L1-to-XBAR read-port width", "read", ("RANGE", "CENSORED"), ("THROUGHPUT_PLATEAU", "RTT_CONTRAST")),
        _quantity_row("b.l1_xbar_write_port_width", "b2", "b", "sim.l1_xbar.write_port_width", "B/SM-cycle", "RTT-invariant per-SM L1-to-XBAR write-port width", "write", ("RANGE", "CENSORED"), ("THROUGHPUT_PLATEAU", "RTT_CONTRAST")),
        _quantity_row("a.l2_dram_fetch_granule", "a1", "a", "sim.l2.dram_fetch_granule", "bytes", "L2-to-DRAM demand-fill granule", "ld.cg", ("RANGE", "MIXTURE"), ("CLOCK64_HIT_MISS_CLASSIFIER", "PERSISTENCE_DELTA_CONTRAST")),
        _quantity_row("a.l2_pair_fill_probability", "a2", "a", "sim.l2.pair_fill_probability", "probability", "Conditioned partner-sector fill probability", "ld.cg", ("RANGE", "MIXTURE"), ("CLOCK64_HIT_MISS_CLASSIFIER", "COMPOSITION_IDENTITY")),
        _quantity_row("d.mshr_sector_capacity", "d1", "d", "sim.mshr.sector_capacity", "sectors", "Per-SM outstanding miss capacity in sectors", "sector", ("CENSORED", "SURFACE_GAP"), ("LITTLE_LAW_TRIANGULATION", "RTT_CONTRAST")),
        _quantity_row("d.mshr_line_capacity", "d2", "d", "sim.mshr.line_capacity", "lines", "Per-SM outstanding miss capacity in lines", "line", ("CENSORED", "SURFACE_GAP"), ("LITTLE_LAW_TRIANGULATION", "RTT_CONTRAST")),
        _quantity_row("f.issue_cost_cycles", "f1", "f", "sim.cp_async.issue_cycles", "cycles", "cp.async issue occupancy per instruction", "cp.async.issue", ("RANGE", "CENSORED"), ("CLOCK64_ELAPSED_WINDOW", "BURST_SLOPE_BREAK")),
        _quantity_row("f.completion_latency_cycles", "f2", "f", "sim.cp_async.completion_latency", "cycles", "cp.async completion latency to consumable shared data", "cp.async.complete", ("RANGE", "MIXTURE"), clock_rtt),
        _quantity_row("f.inflight_capacity_bytes", "f3", "f", "sim.cp_async.inflight_capacity", "bytes", "cp.async in-flight byte capacity", "cp.async.capacity", ("CENSORED", "SURFACE_GAP"), ("LITTLE_LAW_TRIANGULATION", "WAIT_BAR_CONTRAST"), agreement_limit=0.15),
        _quantity_row("f.streaming_bandwidth", "f4", "f", "sim.cp_async.streaming_bandwidth", "B/SM-cycle", "Sustained cp.async streaming bandwidth", "cp.async.stream", ("RANGE", "CENSORED", "UNRESOLVED_STRUCTURE"), ("THROUGHPUT_PLATEAU", "HYBRID_CONTENTION_CONTRAST")),
        _quantity_row("f.wait_group_delta_cycles", "f5", "f", "sim.cp_async.wait_group_delta", "cycles", "Conditioned cp.async wait-group and barrier delay", "cp.async.wait", ("RANGE", "SURFACE_GAP"), ("WAIT_BAR_CONTRAST", "CLOCK64_ELAPSED_WINDOW")),
        _quantity_row("c.dram_read_bandwidth", "c1", "c", "sim.dram.read_bandwidth", "B/SM-cycle", "Locked-point achievable aggregate DRAM read bandwidth", "read", ("RANGE",), throughput),
        _quantity_row("c.dram_write_bandwidth", "c2", "c", "sim.dram.write_bandwidth", "B/SM-cycle", "Locked-point achievable aggregate DRAM write bandwidth", "write", ("RANGE",), throughput),
        _quantity_row("c.dram_copy_bandwidth", "c3", "c", "sim.dram.copy_bandwidth", "B/SM-cycle", "Locked-point achievable aggregate DRAM copy bandwidth", "copy", ("RANGE",), throughput),
        _quantity_row("e.l2_associativity_ways", "e1", "e", "sim.l2.associativity", "ways", "L2 eviction-set associativity", "eviction_set", ("RANGE",), ("CLOCK64_HIT_MISS_CLASSIFIER", "COMPOSITION_IDENTITY")),
        _quantity_row("e.l2_slice_hash_structure", "e2", "e", "sim.l2.slice_hash", "structure", "Resolved low-bit domain and unresolved portion of the L2 slice hash", "slice_hash", ("UNRESOLVED_STRUCTURE",), ("CLOCK64_HIT_MISS_CLASSIFIER", "HYBRID_CONTENTION_CONTRAST")),
        _quantity_row("e.l2_replacement_policy", "e3", "e", "sim.l2.replacement_policy", "structure", "L2 replacement survival-curve policy structure", "replacement", ("UNRESOLVED_STRUCTURE", "RANGE"), ("CLOCK64_HIT_MISS_CLASSIFIER", "COMPOSITION_IDENTITY")),
    ]
    pchase_specs = (
        ("dram_latency_cycles", "cycles_per_dependent_load", "sim.memory.dram_latency", ("RANGE", "CENSORED")),
        ("l1_associativity_ways", "ways", "sim.l1.associativity", ("RANGE", "UNRESOLVED_STRUCTURE")),
        ("l1_capacity_bytes", "bytes", "sim.l1.capacity", ("RANGE",)),
        ("l1_capacity_carveout_0_pct_bytes", "bytes", "sim.l1.capacity.carveout_0", ("RANGE",)),
        ("l1_capacity_carveout_25_pct_bytes", "bytes", "sim.l1.capacity.carveout_25", ("RANGE",)),
        ("l1_capacity_carveout_50_pct_bytes", "bytes", "sim.l1.capacity.carveout_50", ("RANGE",)),
        ("l1_capacity_carveout_75_pct_bytes", "bytes", "sim.l1.capacity.carveout_75", ("RANGE",)),
        ("l1_capacity_carveout_100_pct_bytes", "bytes", "sim.l1.capacity.carveout_100", ("RANGE",)),
        ("l1_hit_latency_cycles", "cycles_per_dependent_load", "sim.memory.l1_hit_latency", ("RANGE",)),
        ("l1_replacement_policy", "structure", "sim.l1.replacement_policy", ("UNRESOLVED_STRUCTURE",)),
        ("l2_capacity_bytes", "bytes", "sim.l2.capacity", ("RANGE", "CENSORED")),
        ("l2_hit_latency_cycles", "cycles_per_dependent_load", "sim.memory.l2_hit_latency", ("RANGE",)),
        ("mshr_outstanding_misses", "entries", "sim.mshr.legacy_outstanding_bound", ("CENSORED", "SURFACE_GAP")),
        ("shared_latency_cycles", "cycles_per_dependent_load", "sim.memory.shared_latency", ("RANGE",)),
    )
    for index, (name, unit, binding, gaps) in enumerate(pchase_specs, start=1):
        rows.append(
            _quantity_row(
                f"pchase.{name}",
                f"pchase{index}",
                None,
                binding,
                unit,
                f"Fresh locked pchase property {name}",
                name,
                gaps,
                clock_rtt if "latency" in name else clock_control,
            )
        )
    return {
        "schema_version": 1,
        "artifact_kind": "M5_QUANTITY_AUTHORITY",
        "authority_id": "sm86-m5-quantity-authority-v1",
        "execution_order": list(EXECUTION_ORDER),
        "scientific_rank": list(SCIENTIFIC_RANK),
        "quantities": rows,
    }


def _authority_unsigned_payload(authority: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(authority))
    unsigned.pop("signature_ed25519", None)
    return unsigned


def _authority_witness(
    raw: bytes, reference: Mapping[str, Any], label: str
) -> dict[str, Any]:
    try:
        value = immutable.strict_json_loads(raw, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    witness = _exact_object(value, label)
    if raw != immutable.pretty_json_bytes(witness):
        _fail(f"{label} is not exact pretty JSON")
    return witness


def _validate_v2_quantity_authority_schema(
    authority: Mapping[str, Any], schema_payload: bytes | None = None
) -> None:
    if schema_payload is None:
        try:
            payload = immutable.read_regular(AUTHORITY_SCHEMA_PATH_V2)
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
    else:
        payload = schema_payload
    if immutable.sha256_bytes(payload) != AUTHORITY_SCHEMA_SHA256_V2:
        _fail("M5 v2 quantity authority schema digest is not controller-pinned")
    schema = _schema_from_bytes(payload, AUTHORITY_SCHEMA_PATH_V2.name)
    if schema.get("$id") != AUTHORITY_SCHEMA_ID_V2:
        _fail("M5 v2 quantity authority schema identifier is not controller-pinned")
    errors = sorted(
        StrictDraft202012Validator(schema).iter_errors(authority),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        _fail(f"M5 v2 quantity authority schema violation at {location}: {error.message}")


def _validate_ancestry_audit_schema(
    audit: Mapping[str, Any], schema_payload: bytes
) -> None:
    if immutable.sha256_bytes(schema_payload) != AUTHORITY_ANCESTRY_AUDIT_SCHEMA_SHA256_V1:
        _fail("M5 ancestry-audit schema digest is not controller-pinned")
    schema = _schema_from_bytes(
        schema_payload, AUTHORITY_ANCESTRY_AUDIT_SCHEMA_PATH_V1.name
    )
    if schema.get("$id") != AUTHORITY_ANCESTRY_AUDIT_SCHEMA_ID_V1:
        _fail("M5 ancestry-audit schema identifier is not controller-pinned")
    errors = sorted(
        StrictDraft202012Validator(schema).iter_errors(audit),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        _fail(f"M5 ancestry-audit schema violation at {location}: {error.message}")


def _retained_ancestry_audit_module(payload: bytes) -> types.ModuleType:
    try:
        source = payload.decode("utf-8", errors="strict")
        code = compile(
            source,
            str(AUTHORITY_ANCESTRY_AUDIT_TOOL_PATH_V1),
            "exec",
            dont_inherit=True,
            optimize=0,
        )
        module = types.ModuleType("_m5_retained_authority_ancestry_audit")
        module.__file__ = str(AUTHORITY_ANCESTRY_AUDIT_TOOL_PATH_V1)
        exec(code, module.__dict__)
        identifiers = (
            getattr(module, "AUDIT_ID", None),
            getattr(module, "AUDIT_MEDIA_TYPE", None),
            getattr(module, "AUDIT_POLICY_VERSION", None),
            getattr(module, "SOURCE_REGISTRY_SCHEMA_ID", None),
        )
        if identifiers != (
            AUTHORITY_ANCESTRY_AUDIT_ID_V1,
            AUTHORITY_ANCESTRY_AUDIT_MEDIA_TYPE_V1,
            1,
            M5_SOURCE_REGISTRY_SCHEMA_ID_V1,
        ):
            _fail("retained M5 ancestry-audit tool contract identifiers changed")
        bind_execution = getattr(module, "bind_executing_tool_sha256", None)
        if not callable(bind_execution) or not callable(
            getattr(module, "build_audit", None)
        ):
            _fail("retained M5 ancestry-audit tool entry point is absent")
        bind_execution(immutable.sha256_bytes(payload))
    except M5ContractError:
        raise
    except Exception as exc:
        _fail(f"retained M5 ancestry-audit tool cannot execute: {exc}")
    return module


def _build_ancestry_audit(
    *,
    authority: Mapping[str, Any],
    authority_raw: bytes,
    design_inputs: Mapping[str, tuple[Mapping[str, Any], bytes]],
    source_registry_payload: bytes,
    source_registry_schema_payload: bytes,
    audit_tool_payload: bytes,
    schema_payload: bytes,
) -> tuple[dict[str, Any], bytes]:
    try:
        executing_tool_payload = immutable.read_regular(
            AUTHORITY_ANCESTRY_AUDIT_TOOL_PATH_V1
        )
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    if (
        immutable.sha256_bytes(executing_tool_payload)
        != AUTHORITY_ANCESTRY_AUDIT_TOOL_SHA256_V1
        or audit_tool_payload != executing_tool_payload
    ):
        _fail("M5 ancestry-audit executing tool differs from retained pinned bytes")
    if (
        immutable.sha256_bytes(source_registry_schema_payload)
        != M5_SOURCE_REGISTRY_SCHEMA_SHA256_V1
    ):
        _fail("M5 source-surface registry schema digest is not controller-pinned")
    audit_implementation = _retained_ancestry_audit_module(audit_tool_payload)
    try:
        audit = audit_implementation.build_audit(
            authority=authority,
            authority_raw=authority_raw,
            design_inputs=design_inputs,
            source_registry_payload=source_registry_payload,
            source_registry_schema_payload=source_registry_schema_payload,
            audit_schema_payload=schema_payload,
            legacy_authority=_quantity_authority_model(),
        )
    except audit_implementation.AncestryAuditError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"retained M5 ancestry-audit execution failed: {exc}")
    _validate_ancestry_audit_schema(audit, schema_payload)
    payload = immutable.pretty_json_bytes(audit)
    return audit, payload


def _trust_anchor_registry(
    registry_payload: bytes | None = None,
    schema_payload: bytes | None = None,
) -> dict[str, Any]:
    try:
        registry_raw = (
            immutable.read_regular(TRUST_ANCHOR_REGISTRY_PATH_V1)
            if registry_payload is None
            else registry_payload
        )
        schema_raw = (
            immutable.read_regular(TRUST_ANCHOR_SCHEMA_PATH_V1)
            if schema_payload is None
            else schema_payload
        )
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    if immutable.sha256_bytes(registry_raw) != TRUST_ANCHOR_REGISTRY_SHA256_V1:
        _fail("M5 trust-anchor registry digest is not controller-pinned")
    if immutable.sha256_bytes(schema_raw) != TRUST_ANCHOR_SCHEMA_SHA256_V1:
        _fail("M5 trust-anchor registry schema digest is not controller-pinned")
    try:
        value = immutable.strict_json_loads(
            registry_raw, label="M5 trust-anchor registry"
        )
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    registry = _exact_object(value, "M5 trust-anchor registry")
    if registry_raw != immutable.pretty_json_bytes(registry):
        _fail("M5 trust-anchor registry is not exact pretty JSON")
    schema = _schema_from_bytes(schema_raw, TRUST_ANCHOR_SCHEMA_PATH_V1.name)
    if schema.get("$id") != TRUST_ANCHOR_SCHEMA_ID_V1:
        _fail("M5 trust-anchor registry schema identifier is not controller-pinned")
    errors = sorted(
        StrictDraft202012Validator(schema).iter_errors(registry),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        _fail(
            "M5 trust-anchor registry schema violation at "
            f"{location}: {error.message}"
        )
    anchor_ids: set[str] = set()
    bindings: set[bytes] = set()
    for index, raw_entry in enumerate(registry["entries"]):
        entry = _exact_object(raw_entry, f"M5 trust-anchor registry entries[{index}]")
        anchor_id = _safe_id(entry["anchor_id"], f"trust anchor entries[{index}].anchor_id")
        if anchor_id in anchor_ids:
            _fail("M5 trust-anchor registry contains duplicate anchor_id")
        anchor_ids.add(anchor_id)
        binding = copy.deepcopy(entry)
        binding.pop("anchor_id")
        binding_bytes = immutable.canonical_json_bytes(binding)
        if binding_bytes in bindings:
            _fail("M5 trust-anchor registry contains an ambiguous authority binding")
        bindings.add(binding_bytes)
    return copy.deepcopy(registry)


def _require_admitted_v2_authority(
    *,
    public_key: bytes,
    authority_reference: Mapping[str, Any],
    ancestry_reference: Mapping[str, Any],
    ancestry_audit_reference: Mapping[str, Any],
    audit_policy_sha256: str,
    design_input_references: Mapping[str, Mapping[str, Any]],
    review_identity_ids: list[str],
    review_references: list[Mapping[str, Any]],
    operator_reference: Mapping[str, Any],
    registry_payload: bytes | None = None,
    registry_schema_payload: bytes | None = None,
) -> str:
    if len(review_identity_ids) != len(review_references):
        _fail("M5_RULE1_TRUST_ANCHOR_UNADMITTED: review binding is incomplete")
    expected = {
        "authority_domain": "sm86_m5_rule1_quantity_authority_v2",
        "trust_anchor_identity_id": "m5-analysis-authority",
        "quantity_authority_identity_id": QUANTITY_AUTHORITY_ID_V2,
        "algorithm": "Ed25519",
        "public_key_sha256": immutable.sha256_bytes(public_key),
        "authority_sha256": authority_reference["sha256"],
        "ancestry_identity_id": AUTHORITY_ANCESTRY_ID_V2,
        "ancestry_sha256": ancestry_reference["sha256"],
        "ancestry_audit_identity_id": AUTHORITY_ANCESTRY_AUDIT_ID_V1,
        "ancestry_audit_sha256": ancestry_audit_reference["sha256"],
        "audit_policy_sha256": audit_policy_sha256,
        "design_inputs": [
            {
                "identity_id": identity_id,
                "sha256": reference["sha256"],
                "size_bytes": reference["size_bytes"],
                "media_type": reference["media_type"],
            }
            for identity_id, reference in sorted(design_input_references.items())
        ],
        "review_witnesses": [
            {
                "identity_id": identity_id,
                "sha256": reference["sha256"],
            }
            for identity_id, reference in zip(
                review_identity_ids, review_references, strict=True
            )
        ],
        "operator_approval_identity_id": AUTHORITY_OPERATOR_APPROVAL_ID_V2,
        "operator_approval_sha256": operator_reference["sha256"],
        "admission_basis": "EXPLICIT_REVIEW_AND_OPERATOR_ACCEPTANCE",
    }
    registry = _trust_anchor_registry(registry_payload, registry_schema_payload)
    matches = []
    for raw_entry in registry["entries"]:
        entry = copy.deepcopy(_exact_object(raw_entry, "M5 trust-anchor entry"))
        anchor_id = entry.pop("anchor_id")
        if entry == expected:
            matches.append(anchor_id)
    if len(matches) != 1:
        _fail(
            "M5_RULE1_TRUST_ANCHOR_UNADMITTED: exact public authority and "
            "witness binding is not uniquely admitted"
        )
    return matches[0]


def _authority_design_inputs(
    authority: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    artifact_loader: Callable[[Mapping[str, Any]], bytes],
) -> dict[str, tuple[Mapping[str, Any], bytes]]:
    provenance = _exact_object(
        authority.get("provenance"), "M5 v2 authority provenance"
    )
    bindings = _exact_array(
        provenance.get("design_input_bindings"),
        "M5 v2 authority design-input bindings",
    )
    design_inputs: dict[str, tuple[Mapping[str, Any], bytes]] = {}
    retained_byte_bindings: set[tuple[str, int]] = set()
    retained_artifact_ids: set[str] = set()
    retained_paths: set[str] = set()
    for index, raw_binding in enumerate(bindings):
        binding = _exact_object(raw_binding, f"design_input_bindings[{index}]")
        identity_id = _safe_id(
            binding.get("identity_id"), f"design_input_bindings[{index}].identity_id"
        )
        identity = identities.get(identity_id)
        if identity is None or identity.get("identity_kind") != "DESIGN_INPUT":
            _fail(
                "M5_RULE1_PREREGISTRATION_ANCESTRY: signed design input is "
                f"absent or wrong-kind: {identity_id}"
            )
        if identity_id in design_inputs:
            _fail("M5_RULE1_PREREGISTRATION_ANCESTRY: duplicate design input")
        artifact = identity["artifact"]
        byte_binding = (artifact["sha256"], artifact["size_bytes"])
        if (
            byte_binding in retained_byte_bindings
            or artifact["artifact_id"] in retained_artifact_ids
            or artifact["relative_path"] in retained_paths
        ):
            _fail(
                "M5_RULE1_PREREGISTRATION_ANCESTRY: distinct design-input "
                "identities alias retained bytes or artifact references"
            )
        retained_byte_bindings.add(byte_binding)
        retained_artifact_ids.add(artifact["artifact_id"])
        retained_paths.add(artifact["relative_path"])
        design_inputs[identity_id] = (
            artifact,
            artifact_loader(artifact),
        )
    retained_ids = {
        identity_id
        for identity_id, identity in identities.items()
        if identity.get("identity_kind") == "DESIGN_INPUT"
    }
    if retained_ids != set(design_inputs):
        _fail(
            "M5_RULE1_PREREGISTRATION_ANCESTRY: retained and signed design-input "
            "sets differ"
        )
    return design_inputs


def _validate_v2_quantity_authority(
    authority: Mapping[str, Any],
    raw: bytes,
    *,
    authority_reference: Mapping[str, Any],
    ancestry_reference: Mapping[str, Any],
    review_references: list[Mapping[str, Any]],
    operator_reference: Mapping[str, Any],
    ancestry: Mapping[str, Any],
    review_witnesses: list[Mapping[str, Any]],
    operator_approval: Mapping[str, Any],
    trust_anchor: bytes,
    schema_payload: bytes | None = None,
) -> dict[str, Any]:
    """Validate an externally retained, signed, target-free v2 authority."""
    _require_keys(
        authority,
        (
            "schema_version",
            "artifact_kind",
            "authority_id",
            "execution_order",
            "scientific_rank",
            "quantities",
            "consumer_binding_registry",
            "gap_mapping",
            "provenance",
            "signature_ed25519",
        ),
        "M5 v2 quantity authority",
    )
    _validate_v2_quantity_authority_schema(authority, schema_payload)
    if (
        authority["schema_version"] != 2
        or authority["artifact_kind"] != "M5_QUANTITY_AUTHORITY"
        or authority["authority_id"] != "sm86-m5-quantity-authority-v2"
    ):
        _fail("M5_RULE1_PREREGISTRATION_VERSION: v1 quantity authority is not admissible")
    legacy_authority = _quantity_authority_model()
    if {
        "execution_order": authority["execution_order"],
        "scientific_rank": authority["scientific_rank"],
        "quantities": authority["quantities"],
    } == {
        "execution_order": legacy_authority["execution_order"],
        "scientific_rank": legacy_authority["scientific_rank"],
        "quantities": legacy_authority["quantities"],
    }:
        _fail(
            "M5_RULE1_PREREGISTRATION_ANCESTRY: v2 authority is an exact "
            "content rewrap of quarantined v1"
        )
    if raw != immutable.pretty_json_bytes(authority):
        _fail("M5 v2 quantity authority is not exact pretty JSON")
    if authority_reference["sha256"] != immutable.sha256_bytes(raw):
        _fail("M5 v2 quantity authority declaration has a stale digest")
    signature_text = _exact_string(
        authority["signature_ed25519"], "M5 v2 quantity authority signature"
    )
    if re.fullmatch(r"[0-9a-f]{128}", signature_text) is None:
        _fail("M5 v2 quantity authority signature is not canonical hex")
    _verify_ed25519(
        trust_anchor,
        _AUTHORITY_SIGNATURE_DOMAIN
        + immutable.canonical_json_bytes(_authority_unsigned_payload(authority)),
        bytes.fromhex(signature_text),
    )
    if len(authority["execution_order"]) != len(set(authority["execution_order"])):
        _fail("M5 v2 authority execution order contains duplicate families")
    if len(authority["scientific_rank"]) != len(set(authority["scientific_rank"])):
        _fail("M5 v2 authority scientific rank contains duplicate families")
    if set(authority["execution_order"]) != set(authority["scientific_rank"]):
        _fail("M5 v2 authority family orders cover different families")
    expected_registry = [
        {
            "quantity_id": quantity["quantity_id"],
            "consumer_binding": quantity["consumer_binding"],
            "authorized_gap_kinds": list(quantity["authorized_gap_kinds"]),
            "research_gap_only": True,
        }
        for quantity in authority["quantities"]
    ]
    if authority["consumer_binding_registry"] != expected_registry:
        _fail("M5_RULE1_CONSUMER_BINDING_REGISTRY: registry is not authority-derived")
    if authority["gap_mapping"] != {
        "mapping_version": 1,
        "typed_gap_kinds": sorted(TYPED_GAPS),
        "consumer_rule": "RETAIN_AS_TYPED_RESEARCH_GAP",
        "ncu_selection_forbidden": True,
    }:
        _fail("M5_RULE1_GAP_MAPPING: consumer gap mapping is not fail-closed")
    provenance = _exact_object(authority["provenance"], "M5 v2 authority provenance")
    _require_keys(
        provenance,
        (
            "ancestry_identity_id",
            "review_identity_ids",
            "operator_approval_identity_id",
            "trust_anchor_identity_id",
            "target_free",
            "design_input_bindings",
            "scientific_field_sources",
        ),
        "M5 v2 authority provenance",
    )
    if (
        provenance["ancestry_identity_id"] != AUTHORITY_ANCESTRY_ID_V2
        or type(provenance["review_identity_ids"]) is not list
        or len(provenance["review_identity_ids"]) != len(review_references)
        or any(type(identity_id) is not str for identity_id in provenance["review_identity_ids"])
        or provenance["operator_approval_identity_id"]
        != AUTHORITY_OPERATOR_APPROVAL_ID_V2
        or provenance["trust_anchor_identity_id"] != "m5-analysis-authority"
        or provenance["target_free"] is not True
    ):
        _fail("M5_RULE1_PREREGISTRATION_ANCESTRY: authority provenance is not closed")
    _require_keys(
        ancestry,
        (
            "schema_version",
            "artifact_kind",
            "authority_sha256",
            "target_free",
            "content_informing_ancestor_ids",
            "forbidden_source_classes",
        ),
        "M5 authority ancestry witness",
    )
    if (
        ancestry["schema_version"] != 1
        or ancestry["artifact_kind"] != "M5_AUTHORITY_ANCESTRY"
        or ancestry["authority_sha256"] != authority_reference["sha256"]
        or ancestry["target_free"] is not True
        or ancestry["content_informing_ancestor_ids"] != []
        or ancestry["forbidden_source_classes"]
        != ["NCU_TARGET", "SIMULATOR_ERROR", "TARGET_BRACKET", "DERIVED_SELECTION"]
    ):
        _fail("M5_RULE1_PREREGISTRATION_ANCESTRY: target-bearing ancestry witness")
    if len(review_witnesses) != len(review_references) or not review_witnesses:
        _fail("M5_RULE1_INDEPENDENT_REVIEW_MISSING: review witness coverage is incomplete")
    for index, (review, reference) in enumerate(zip(review_witnesses, review_references, strict=True)):
        _require_keys(
            review,
            (
                "schema_version",
                "artifact_kind",
                "authority_sha256",
                "reviewer_id",
                "blind_review",
                "independent",
                "approved",
            ),
            f"M5 authority review witness[{index}]",
        )
        if (
            review["schema_version"] != 1
            or review["artifact_kind"] != "M5_AUTHORITY_BLIND_REVIEW"
            or review["authority_sha256"] != authority_reference["sha256"]
            or not review["reviewer_id"]
            or review["blind_review"] is not True
            or review["independent"] is not True
            or review["approved"] is not True
            or reference["sha256"] != immutable.sha256_bytes(
                immutable.pretty_json_bytes(review)
            )
        ):
            _fail("M5_RULE1_INDEPENDENT_REVIEW_MISSING: review witness is not independent")
    _require_keys(
        operator_approval,
        (
            "schema_version",
            "artifact_kind",
            "authority_sha256",
            "operator_id",
            "approved",
        ),
        "M5 authority operator approval",
    )
    if (
        operator_approval["schema_version"] != 1
        or operator_approval["artifact_kind"] != "M5_AUTHORITY_OPERATOR_APPROVAL"
        or operator_approval["authority_sha256"] != authority_reference["sha256"]
        or not operator_approval["operator_id"]
        or operator_approval["approved"] is not True
        or operator_reference["sha256"]
        != immutable.sha256_bytes(immutable.pretty_json_bytes(operator_approval))
    ):
        _fail("M5_RULE1_OPERATOR_APPROVAL_MISSING: operator witness is not closed")
    if ancestry_reference["sha256"] != immutable.sha256_bytes(
        immutable.pretty_json_bytes(ancestry)
    ):
        _fail("M5 authority ancestry declaration has a stale digest")
    return copy.deepcopy(dict(authority))


def _strict_integer(_checker: Any, instance: Any) -> bool:
    return type(instance) is int and abs(instance) <= _MAX_SAFE_JSON_INTEGER


def _strict_number(_checker: Any, instance: Any) -> bool:
    return (
        type(instance) is int
        and abs(instance) <= _MAX_SAFE_JSON_INTEGER
    ) or (type(instance) is float and math.isfinite(instance))


_STRICT_TYPE_CHECKER = (
    jsonschema.Draft202012Validator.TYPE_CHECKER.redefine("integer", _strict_integer).redefine(
        "number", _strict_number
    )
)
StrictDraft202012Validator = validators.extend(
    jsonschema.Draft202012Validator, type_checker=_STRICT_TYPE_CHECKER
)


class M5ContractError(RuntimeError):
    """A versioned M5 contract, policy, or immutable-byte check failed."""


def _fail(message: str) -> None:
    raise M5ContractError(message)


def _contract_boundary(function: Callable[..., Any]) -> Callable[..., Any]:
    """Convert malformed external structure and filesystem failures to one typed refusal."""

    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except M5ContractError:
            raise
        except immutable.ImmutableArtifactError as exc:
            raise M5ContractError(str(exc)) from exc
        except (KeyError, TypeError, IndexError, ValueError, OverflowError, OSError) as exc:
            raise M5ContractError(f"malformed or inaccessible M5 contract: {exc}") from exc

    return guarded


def _exact_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact JSON object")
    return value


def _exact_array(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label} must be an exact JSON array")
    return value


def _exact_string(value: Any, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be an exact JSON string")
    return value


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int or abs(value) > _MAX_SAFE_JSON_INTEGER:
        _fail(f"{label} must be an exact safe-range JSON integer")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be an exact JSON boolean")
    return value


def _require_keys(value: Mapping[str, Any], required: Iterable[str], label: str) -> None:
    expected = set(required)
    observed = set(value)
    if observed != expected:
        _fail(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _safe_id(value: Any, label: str) -> str:
    text = _exact_string(value, label)
    if _SAFE_ID.fullmatch(text) is None:
        _fail(f"{label} is not a safe identifier")
    return text


def _schema_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        schema = immutable.strict_json_loads(payload, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    schema = _exact_object(schema, label)
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        _fail(f"{label} is not Draft-2020-12")

    def reject_external_references(node: Any) -> None:
        if type(node) is dict:
            for key, value in node.items():
                if key in {"$ref", "$dynamicRef"} and (
                    type(value) is not str or not value.startswith("#")
                ):
                    _fail(f"{label} contains a non-bundled schema reference")
                reject_external_references(value)
        elif type(node) is list:
            for value in node:
                reject_external_references(value)

    reject_external_references(schema)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        _fail(f"invalid schema authority {label}: {exc.message}")
    return schema


def _schema_version_for_model(model: Mapping[str, Any], kind: str) -> int:
    if model.get("artifact_kind") != kind or type(model.get("schema_version")) is not int:
        _fail(f"unsupported explicit schema dispatch for {kind}")
    version = model["schema_version"]
    if version not in SCHEMA_PATHS_BY_VERSION:
        _fail(f"unsupported explicit schema dispatch: {kind} v{version}")
    return version


def _live_schema(kind: str, version: int = ACTIVE_SCHEMA_VERSION) -> dict[str, Any]:
    path = SCHEMA_PATHS_BY_VERSION.get(version, {}).get(kind)
    if path is None:
        _fail(f"unsupported explicit schema dispatch: {kind} v{version}")
    try:
        payload = immutable.read_regular(path)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    expected_digest = SCHEMA_SHA256_BY_VERSION[version][kind]
    if immutable.sha256_bytes(payload) != expected_digest:
        _fail(f"live {kind} schema digest is not controller-pinned v{version}")
    schema = _schema_from_bytes(payload, path.name)
    if schema.get("$id") != SCHEMA_IDS_BY_VERSION[version][kind]:
        _fail(f"live {kind} schema has the wrong explicit v{version} schema identifier")
    return schema


def _validate_schema(model: Mapping[str, Any], kind: str, schema: Mapping[str, Any]) -> None:
    version = _schema_version_for_model(model, kind)
    expected_id = SCHEMA_IDS_BY_VERSION[version][kind]
    if schema.get("$id") != expected_id:
        _fail(f"schema argument is not the selected {kind} v{version} authority")
    errors = sorted(
        StrictDraft202012Validator(schema).iter_errors(model),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        _fail(f"{kind} schema violation at {location}: {error.message}")


def _verify_sealed_model(model: Mapping[str, Any], kind: str, schema: Mapping[str, Any]) -> None:
    _validate_schema(model, kind, schema)
    try:
        immutable.verify_model_seal(model)
    except immutable.ImmutableArtifactError as exc:
        _fail(f"{kind} canonical model seal: {exc}")


def _load_json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = immutable.read_regular(path)
        model = immutable.strict_json_loads(raw, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    return _exact_object(model, label), raw


def _load_json_relative(root: Path, relative_path: str, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = immutable.read_relative(root, relative_path)
        model = immutable.strict_json_loads(raw, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    return _exact_object(model, label), raw


def _require_pretty_bytes(model: Mapping[str, Any], raw: bytes, label: str) -> None:
    if raw != immutable.pretty_json_bytes(model):
        _fail(f"{label} bytes are not the exact pretty serialization")


def _contains_card_path(path: Path) -> bool:
    for parts in (
        Path(os.path.abspath(os.fspath(path))).parts,
        Path(os.path.realpath(os.fspath(path))).parts,
    ):
        for index in range(len(parts) - 1):
            if parts[index : index + 2] == ("arch_profiles", "cards"):
                return True
    return False


def _refuse_card_path(path: Path, label: str) -> None:
    if _contains_card_path(path):
        _fail(f"architecture-card path refused for {label}")


def _external_output_root(path: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    _refuse_card_path(root, "publication output")
    repository = Path(os.path.realpath(REPO_ROOT))
    candidate = Path(os.path.realpath(root))
    if os.path.commonpath((os.fspath(repository), os.fspath(candidate))) == os.fspath(repository):
        _fail("immutable M5 publication root must be external to the repository")
    return root


def _directory_ancestor_identities(descriptor: int) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    current = os.dup(descriptor)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for _ in range(1024):
            metadata = os.fstat(current)
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in identities:
                _fail("directory ancestry contains a cycle")
            identities.add(identity)
            parent = os.open("..", flags, dir_fd=current)
            parent_metadata = os.fstat(parent)
            if (parent_metadata.st_dev, parent_metadata.st_ino) == identity:
                os.close(parent)
                return identities
            os.close(current)
            current = parent
    finally:
        os.close(current)
    _fail("directory ancestry exceeds the bounded traversal limit")


def _authorize_output_descriptor(root: Path, descriptor: int) -> None:
    absolute = Path(os.path.abspath(os.fspath(root)))
    try:
        linked = os.stat(absolute, follow_symlinks=False)
        opened = os.fstat(descriptor)
        descriptor_target = os.readlink(f"/proc/self/fd/{descriptor}")
    except OSError as exc:
        _fail(f"cannot reauthorize publication root at commit: {exc}")
    if (
        stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        _fail("publication root identity changed before commit")
    if not os.path.isabs(descriptor_target) or descriptor_target.endswith(" (deleted)"):
        _fail("publication root descriptor has no stable named path")
    named_real = Path(os.path.realpath(absolute))
    descriptor_real = Path(os.path.realpath(descriptor_target))
    if named_real != descriptor_real:
        _fail("publication root descriptor no longer names the authorized root")
    _refuse_card_path(absolute, "publication output commit")
    _refuse_card_path(descriptor_real, "publication output descriptor")
    repository = Path(os.path.realpath(REPO_ROOT))
    if os.path.commonpath((os.fspath(repository), os.fspath(descriptor_real))) == os.fspath(
        repository
    ):
        _fail("immutable M5 publication root became repository-local")
    repository_metadata = os.stat(repository, follow_symlinks=False)
    if (repository_metadata.st_dev, repository_metadata.st_ino) in _directory_ancestor_identities(
        descriptor
    ):
        _fail("immutable M5 publication root is within the repository inode tree")


def _require_output_disjoint_from_run(output_root: Path, run_root: Path) -> None:
    output = Path(os.path.realpath(output_root))
    run = Path(os.path.realpath(run_root))
    common = os.path.commonpath((os.fspath(output), os.fspath(run)))
    if common in {os.fspath(output), os.fspath(run)}:
        _fail("analysis output root and RUN bundle must be fully disjoint")


def _resolve_input_path(base: Path, value: Any, label: str) -> Path:
    candidate = Path(_exact_string(value, label))
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    _refuse_card_path(candidate, label)
    return candidate


def _artifact_storage_path(artifact_id: str, digest: str) -> str:
    return f"artifacts/{digest}/{artifact_id}"


def _artifact_reference(
    artifact_id: str, digest: str, size: int, media_type: str
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "relative_path": _artifact_storage_path(artifact_id, digest),
        "sha256": digest,
        "size_bytes": size,
        "media_type": media_type,
    }


def _artifact_input(
    value: Any, *, base: Path, label: str
) -> tuple[dict[str, Any], immutable.ImmutableFile]:
    item = _exact_object(value, label)
    _require_keys(item, ("artifact_id", "path", "media_type"), label)
    artifact_id = _safe_id(item["artifact_id"], f"{label}.artifact_id")
    media_type = _exact_string(item["media_type"], f"{label}.media_type")
    if not media_type:
        _fail(f"{label}.media_type must not be empty")
    source = _resolve_input_path(base, item["path"], f"{label}.path")
    try:
        digest, size = immutable.sha256_file(source)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    reference = _artifact_reference(artifact_id, digest, size, media_type)
    return reference, immutable.ImmutableFile(
        reference["relative_path"], digest, size, source_path=source
    )


def _artifact_from_path(
    identity_id: str,
    kind: str,
    version: str,
    path: Path,
    media_type: str,
) -> tuple[dict[str, Any], immutable.ImmutableFile]:
    try:
        digest, size = immutable.sha256_file(path)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    artifact_id = f"{identity_id}-artifact"
    reference = _artifact_reference(artifact_id, digest, size, media_type)
    identity = {
        "identity_id": identity_id,
        "identity_kind": kind,
        "version": version,
        "artifact": reference,
    }
    return identity, immutable.ImmutableFile(
        reference["relative_path"], digest, size, source_path=path
    )


def _artifact_from_bytes(
    identity_id: str,
    kind: str,
    version: str,
    payload: bytes,
    media_type: str,
) -> tuple[dict[str, Any], immutable.ImmutableFile]:
    digest = immutable.sha256_bytes(payload)
    artifact_id = f"{identity_id}-artifact"
    reference = _artifact_reference(artifact_id, digest, len(payload), media_type)
    identity = {
        "identity_id": identity_id,
        "identity_kind": kind,
        "version": version,
        "artifact": reference,
    }
    return identity, immutable.ImmutableFile.from_bytes(reference["relative_path"], payload)


def _merge_material(
    destination: dict[str, immutable.ImmutableFile], material: immutable.ImmutableFile
) -> None:
    previous = destination.get(material.relative_path)
    if previous is None:
        destination[material.relative_path] = material
    elif previous.sha256 != material.sha256 or previous.size_bytes != material.size_bytes:
        _fail(f"artifact publication collision: {material.relative_path}")


def _material_bytes(material: immutable.ImmutableFile) -> bytes:
    if material.payload is not None:
        payload = material.payload
    else:
        assert material.source_path is not None
        try:
            payload = immutable.read_regular(material.source_path)
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
    if len(payload) != material.size_bytes or immutable.sha256_bytes(payload) != material.sha256:
        _fail(f"source-backed material changed: {material.relative_path}")
    return payload


class _DeclaredSnapshotLoader:
    """Memoize stable bytes and bind them immediately to each declaration."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        material: Mapping[str, immutable.ImmutableFile] | None = None,
    ) -> None:
        if (root is None) == (material is None):
            _fail("snapshot loader requires exactly one byte source")
        self._root = root
        self._reader = immutable.PinnedDirectoryReader(root) if root is not None else None
        self._material = material
        self._cache: dict[str, tuple[str, int, bytes]] = {}
        self.frozen_files: dict[str, immutable.ImmutableFile] = {}

    def __call__(self, reference: Mapping[str, Any]) -> bytes:
        artifact_id = _safe_id(reference.get("artifact_id"), "artifact artifact_id")
        relative_path = _exact_string(
            reference.get("relative_path"), "artifact relative_path"
        )
        try:
            immutable.safe_relative_path(relative_path)
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
        digest = _exact_string(reference.get("sha256"), "artifact sha256")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            _fail("artifact sha256 must be lowercase 64-hex")
        if relative_path != _artifact_storage_path(artifact_id, digest):
            _fail("artifact relative_path is not its canonical content-addressed path")
        size = _exact_int(reference.get("size_bytes"), "artifact size_bytes")
        if size < 0:
            _fail("artifact size_bytes must be nonnegative")
        cached = self._cache.get(relative_path)
        if cached is not None:
            cached_digest, cached_size, payload = cached
            if (digest, size) != (cached_digest, cached_size):
                _fail(f"conflicting declarations for memoized artifact {relative_path}")
            return payload
        if self._material is not None:
            source = self._material.get(relative_path)
            if source is None:
                _fail(f"artifact has no publication material: {relative_path}")
            if (source.sha256, source.size_bytes) != (digest, size):
                _fail(f"artifact declaration differs from publication material: {relative_path}")
            payload = _material_bytes(source)
        else:
            assert self._reader is not None
            try:
                payload = self._reader.read(relative_path)
            except immutable.ImmutableArtifactError as exc:
                _fail(str(exc))
        if len(payload) != size or immutable.sha256_bytes(payload) != digest:
            _fail(f"artifact bytes do not match their declaration: {relative_path}")
        self._cache[relative_path] = (digest, size, payload)
        self.frozen_files[relative_path] = immutable.ImmutableFile.from_bytes(
            relative_path, payload
        )
        return payload

    def read_manifest(self, relative_path: str, label: str) -> tuple[dict[str, Any], bytes]:
        if self._reader is None:
            _fail("manifest reads require a pinned bundle root")
        try:
            payload = self._reader.read(relative_path)
            value = immutable.strict_json_loads(payload, label=label)
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
        return _exact_object(value, label), payload

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def verify_root(self) -> None:
        if self._reader is not None:
            try:
                self._reader.verify_path()
            except immutable.ImmutableArtifactError as exc:
                _fail(str(exc))

    def __enter__(self) -> "_DeclaredSnapshotLoader":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def freeze_into(self, destination: dict[str, immutable.ImmutableFile]) -> None:
        for relative_path, frozen in self.frozen_files.items():
            previous = destination.get(relative_path)
            if previous is None or (
                previous.sha256,
                previous.size_bytes,
            ) != (frozen.sha256, frozen.size_bytes):
                _fail(f"cannot freeze undeclared material: {relative_path}")
            destination[relative_path] = frozen


class _AuthenticationLedger:
    """Authenticate every external analysis artifact against the RUN trust key."""

    def __init__(
        self,
        run: Mapping[str, Any],
        *,
        base: Path,
        material: dict[str, immutable.ImmutableFile],
        loader: _DeclaredSnapshotLoader,
        run_artifact_loader: Callable[[Mapping[str, Any]], bytes],
    ) -> None:
        self._run = run
        self._base = base
        self._material = material
        self._loader = loader
        identities = _identity_map(run)
        authority = identities.get("m5-analysis-authority")
        if authority is None:
            _fail("RUN has no analysis authority")
        self.authority_identity_id = "m5-analysis-authority"
        self.public_key_sha256 = authority["artifact"]["sha256"]
        self._public_key = run_artifact_loader(authority["artifact"])
        self.entries: list[dict[str, Any]] = []
        self._receipt_ids: set[str] = set()

    def add(self, value: Any, *, role: str, context_id: str, label: str) -> dict[str, Any]:
        authenticated = _exact_object(value, label)
        _require_keys(authenticated, ("artifact_input", "receipt_input"), label)
        reference, file = _artifact_input(
            authenticated["artifact_input"], base=self._base, label=f"{label}.artifact_input"
        )
        _merge_material(self._material, file)
        receipt_reference, receipt_file = _artifact_input(
            authenticated["receipt_input"], base=self._base, label=f"{label}.receipt_input"
        )
        _merge_material(self._material, receipt_file)
        if receipt_reference["media_type"] != "application/json":
            _fail(f"{label} authority receipt must use application/json")
        receipt_raw = self._loader(receipt_reference)
        try:
            receipt = immutable.strict_json_loads(receipt_raw, label=f"{label} authority receipt")
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
        receipt = _exact_object(receipt, f"{label} authority receipt")
        _require_keys(
            receipt,
            (
                "schema_version",
                "artifact_kind",
                "authority_identity_id",
                "run_model_seal_sha256",
                "source_commit",
                "artifact_id",
                "artifact_sha256",
                "artifact_size_bytes",
                "artifact_media_type",
                "artifact_role",
                "context_id",
                "signature_ed25519",
            ),
            f"{label} authority receipt",
        )
        if receipt_raw != immutable.pretty_json_bytes(receipt):
            _fail(f"{label} authority receipt is not exact pretty JSON")
        expected = {
            "schema_version": 1,
            "artifact_kind": "M5_AUTHORITY_RECEIPT",
            "authority_identity_id": self.authority_identity_id,
            "run_model_seal_sha256": self._run["model_seal_sha256"],
            "source_commit": self._run["source_state"]["source_commit"],
            "artifact_id": reference["artifact_id"],
            "artifact_sha256": reference["sha256"],
            "artifact_size_bytes": reference["size_bytes"],
            "artifact_media_type": reference["media_type"],
            "artifact_role": role,
            "context_id": context_id,
        }
        unsigned = dict(receipt)
        signature_hex = unsigned.pop("signature_ed25519")
        if unsigned != expected:
            _fail(f"{label} authority receipt declaration is stale or cross-context")
        if type(signature_hex) is not str or re.fullmatch(r"[0-9a-f]{128}", signature_hex) is None:
            _fail(f"{label} authority receipt signature is not lowercase 128-hex")
        _verify_ed25519(
            self._public_key,
            _RECEIPT_DOMAIN + immutable.canonical_json_bytes(unsigned),
            bytes.fromhex(signature_hex),
        )
        if receipt_reference["artifact_id"] in self._receipt_ids:
            _fail("one authority receipt cannot authenticate multiple analysis inputs")
        self._receipt_ids.add(receipt_reference["artifact_id"])
        self.entries.append(
            {
                "artifact": reference,
                "artifact_role": role,
                "context_id": context_id,
                "receipt": receipt_reference,
            }
        )
        return reference

    def model(self) -> dict[str, Any]:
        return {
            "authority_identity_id": self.authority_identity_id,
            "public_key_sha256": self.public_key_sha256,
            "receipt_domain": "M5_AUTHORITY_RECEIPT_V1",
            "entries": sorted(
                self.entries,
                key=lambda entry: (
                    entry["artifact_role"],
                    entry["context_id"],
                    entry["artifact"]["artifact_id"],
                ),
            ),
        }


def _verify_authenticity_ledger(
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, Any]], bytes],
    *,
    expected_public_key_sha256: str | None = None,
) -> None:
    model = _exact_object(result["authenticity"], "RESULT authenticity ledger")
    _require_keys(
        model,
        ("authority_identity_id", "public_key_sha256", "receipt_domain", "entries"),
        "RESULT authenticity ledger",
    )
    identities = _identity_map(run)
    authority = identities.get("m5-analysis-authority")
    if authority is None:
        _fail("authenticity ledger has no RUN authority")
    public_key_reference = authority["artifact"]
    if (
        model["authority_identity_id"] != "m5-analysis-authority"
        or model["public_key_sha256"] != public_key_reference["sha256"]
        or model["receipt_domain"] != "M5_AUTHORITY_RECEIPT_V1"
    ):
        _fail("RESULT authenticity ledger authority binding is stale")
    if expected_public_key_sha256 is not None and model["public_key_sha256"] != expected_public_key_sha256:
        _fail("bundle analysis authority does not match the out-of-band trust anchor")
    public_key = artifact_loader(public_key_reference)
    _validate_ed25519_public_key(public_key, "RESULT authenticity authority")
    entries = _exact_array(model["entries"], "RESULT authenticity entries")
    expected_order = sorted(
        entries,
        key=lambda entry: (
            entry["artifact_role"],
            entry["context_id"],
            entry["artifact"]["artifact_id"],
        ),
    )
    if entries != expected_order:
        _fail("RESULT authenticity entries are not in canonical role/context order")
    receipt_ids: set[str] = set()
    artifact_ids: set[str] = set()
    for index, raw_entry in enumerate(entries):
        label = f"RESULT authenticity entries[{index}]"
        entry = _exact_object(raw_entry, label)
        _require_keys(entry, ("artifact", "artifact_role", "context_id", "receipt"), label)
        artifact = _exact_object(entry["artifact"], f"{label}.artifact")
        receipt_reference = _exact_object(entry["receipt"], f"{label}.receipt")
        artifact_id = _safe_id(artifact.get("artifact_id"), f"{label}.artifact_id")
        receipt_id = _safe_id(receipt_reference.get("artifact_id"), f"{label}.receipt_id")
        if artifact_id in artifact_ids or receipt_id in receipt_ids:
            _fail("RESULT authenticity ledger reuses an artifact or receipt ID")
        artifact_ids.add(artifact_id)
        receipt_ids.add(receipt_id)
        _collect_artifact({}, artifact, f"{label}.artifact")
        _collect_artifact({}, receipt_reference, f"{label}.receipt")
        if receipt_reference["media_type"] != "application/json":
            _fail(f"{label} receipt media type is not application/json")
        receipt_raw = artifact_loader(receipt_reference)
        try:
            receipt = immutable.strict_json_loads(receipt_raw, label=f"{label} receipt")
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
        receipt = _exact_object(receipt, f"{label} receipt")
        _require_keys(
            receipt,
            (
                "schema_version",
                "artifact_kind",
                "authority_identity_id",
                "run_model_seal_sha256",
                "source_commit",
                "artifact_id",
                "artifact_sha256",
                "artifact_size_bytes",
                "artifact_media_type",
                "artifact_role",
                "context_id",
                "signature_ed25519",
            ),
            f"{label} receipt",
        )
        if receipt_raw != immutable.pretty_json_bytes(receipt):
            _fail(f"{label} receipt is not exact pretty JSON")
        unsigned = dict(receipt)
        signature_hex = unsigned.pop("signature_ed25519")
        expected = {
            "schema_version": 1,
            "artifact_kind": "M5_AUTHORITY_RECEIPT",
            "authority_identity_id": "m5-analysis-authority",
            "run_model_seal_sha256": run["model_seal_sha256"],
            "source_commit": run["source_state"]["source_commit"],
            "artifact_id": artifact["artifact_id"],
            "artifact_sha256": artifact["sha256"],
            "artifact_size_bytes": artifact["size_bytes"],
            "artifact_media_type": artifact["media_type"],
            "artifact_role": entry["artifact_role"],
            "context_id": entry["context_id"],
        }
        if unsigned != expected or type(signature_hex) is not str or re.fullmatch(
            r"[0-9a-f]{128}", signature_hex
        ) is None:
            _fail(f"{label} signed declaration is stale or malformed")
        _verify_ed25519(
            public_key,
            _RECEIPT_DOMAIN + immutable.canonical_json_bytes(unsigned),
            bytes.fromhex(signature_hex),
        )
        artifact_loader(artifact)


def _canonical_safe_id(prefix: str, *parts: str) -> str:
    """Map authority text to one collision-resistant lowercase identifier."""
    raw = "\0".join(parts)
    stem = re.sub(r"[^a-z0-9_.-]+", "-", ".".join(parts).lower()).strip("-.")
    if not stem:
        stem = "cell"
    suffix = immutable.sha256_bytes(raw.encode("utf-8"))[:16]
    candidate = f"{prefix}.{stem[:96]}.{suffix}"
    if _SAFE_ID.fullmatch(candidate) is None:
        _fail("canonical authority identifier cannot be represented safely")
    return candidate


def _artifact_signature(reference: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        reference.get(key)
        for key in ("artifact_id", "relative_path", "sha256", "size_bytes", "media_type")
    )


def _collect_artifact(
    by_id: dict[str, dict[str, Any]], reference: Mapping[str, Any], label: str
) -> None:
    artifact_id = _safe_id(reference.get("artifact_id"), f"{label}.artifact_id")
    relative_path = _exact_string(reference.get("relative_path"), f"{label}.relative_path")
    try:
        immutable.safe_relative_path(relative_path)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    digest = _exact_string(reference.get("sha256"), f"{label}.sha256")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        _fail(f"{label}.sha256 must be lowercase 64-hex")
    expected_path = _artifact_storage_path(artifact_id, digest)
    if relative_path != expected_path:
        _fail(f"{label}.relative_path is not the canonical content-addressed path")
    current = dict(reference)
    previous = by_id.get(artifact_id)
    if previous is not None and _artifact_signature(previous) != _artifact_signature(current):
        _fail(f"artifact_id {artifact_id!r} has conflicting byte identities")
    by_id[artifact_id] = current


def _identity_map(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for index, identity in enumerate(run["identities"]):
        identity_id = identity["identity_id"]
        if identity_id in result:
            _fail(f"duplicate identity_id: {identity_id}")
        result[identity_id] = identity
        _collect_artifact(artifacts, identity["artifact"], f"identities[{index}].artifact")
    return result


def _require_identity_kind(
    identities: Mapping[str, Mapping[str, Any]], identity_id: str, expected: str, label: str
) -> None:
    identity = identities.get(identity_id)
    if identity is None:
        _fail(f"{label} references absent identity {identity_id!r}")
    if identity["identity_kind"] != expected:
        _fail(f"{label} must reference {expected}, not {identity['identity_kind']}")


def _git_invocation_environment() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
    }


def _observe_git_source() -> tuple[str, bytes]:
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        _fail("cannot locate Git through the sanitized system path")
    executable = os.path.realpath(executable)
    command = [executable, "-C", os.fspath(REPO_ROOT), "rev-parse", "HEAD"]
    environment = _git_invocation_environment()
    try:
        commit_before = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        ).stdout.decode("ascii").strip()
        status_bytes = subprocess.run(
            [
                executable,
                "-C",
                os.fspath(REPO_ROOT),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        ).stdout
        commit_after = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        ).stdout.decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        _fail(f"cannot derive live Git source observation: {exc}")
    if commit_before != commit_after:
        _fail("Git HEAD changed while observing porcelain-v1-z status")
    return commit_before, status_bytes


def _checked_observation(
    observer: Callable[[], tuple[str, bytes]], label: str
) -> tuple[str, bytes]:
    observation = observer()
    if type(observation) is not tuple or len(observation) != 2:
        _fail(f"{label} source observer returned an invalid contract")
    commit, status_bytes = observation
    if type(commit) is not str or _HEX40.fullmatch(commit) is None:
        _fail(f"{label} source observer returned an invalid full commit")
    if type(status_bytes) is not bytes:
        _fail(f"{label} source observer did not return porcelain-v1-z bytes")
    return commit, status_bytes


def _campaign_protocol_model(payload: bytes) -> dict[str, Any]:
    try:
        model = immutable.strict_json_loads(payload, label="campaign protocol authority")
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    model = _exact_object(model, "campaign protocol authority")
    _require_keys(
        model,
        (
            "schema_version",
            "_note",
            "arch",
            "sku",
            "toolchain",
            "reference_clock_state",
            "measurement_protocol",
            "injection_runtime",
        ),
        "campaign protocol authority",
    )
    toolchain = _exact_object(model["toolchain"], "campaign protocol toolchain")
    _require_keys(
        toolchain,
        ("ncu_version_prefix", "driver_version", "nvcc_release", "nvbit_version", "_nvcc_note"),
        "campaign protocol toolchain",
    )
    clocks = _exact_object(
        model["reference_clock_state"], "campaign protocol reference clock state"
    )
    _require_keys(
        clocks,
        ("sm_clock_mhz", "mem_clock_mhz", "persistence_mode", "_note"),
        "campaign protocol reference clock state",
    )
    measurement = _exact_object(
        model["measurement_protocol"], "campaign protocol measurement protocol"
    )
    _require_keys(
        measurement,
        ("version", "ncu_clock_control", "ncu_cache_control", "ncu_replay_mode"),
        "campaign protocol measurement protocol",
    )
    injection = _exact_object(
        model["injection_runtime"], "campaign protocol injection runtime"
    )
    _require_keys(
        injection,
        (
            "pb_lib_relpath",
            "tracer_so_relpath",
            "nvbit_version_header_relpath",
            "smoke_binary_relpath",
            "smoke_shape_argv",
            "_smoke_note",
        ),
        "campaign protocol injection runtime",
    )
    exact = (
        model["schema_version"] == 1
        and model["arch"] == "sm_86"
        and model["sku"] == "rtx_3090_ti"
        and toolchain["ncu_version_prefix"] == "2026.1.0"
        and toolchain["driver_version"] == "610.43.02"
        and toolchain["nvcc_release"] == "13.3"
        and toolchain["nvbit_version"] == "1.8"
        and clocks["sm_clock_mhz"] == 1350
        and clocks["mem_clock_mhz"] == 10501
        and clocks["persistence_mode"] == "Enabled"
        and measurement
        == {
            "version": "v2.1",
            "ncu_clock_control": "none",
            "ncu_cache_control": "all",
            "ncu_replay_mode": "kernel",
        }
        and injection["pb_lib_relpath"] == ".toolchain/pb/lib"
        and injection["tracer_so_relpath"]
        == "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
        and injection["nvbit_version_header_relpath"]
        == "tools/nvbit_tracer/nvbit_release/core/nvbit.h"
        and injection["smoke_binary_relpath"]
        == "workloads/portable_lcd/silu_fwd_bf16_sm80/silu_fwd_bf16_sm80_run"
        and injection["smoke_shape_argv"] == ["--correctness_shape"]
    )
    if not exact:
        _fail("campaign protocol authority does not contain the complete exact v2.1 sm_86 pin contract")
    return model


def _auto_identities(
    quantities: Iterable[Mapping[str, Any]],
    *,
    schema_version: int = ACTIVE_SCHEMA_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, immutable.ImmutableFile]]:
    identities: list[dict[str, Any]] = []
    material: dict[str, immutable.ImmutableFile] = {}
    schema_paths = SCHEMA_PATHS_BY_VERSION.get(schema_version)
    schema_digests = SCHEMA_SHA256_BY_VERSION.get(schema_version)
    if schema_paths is None or schema_digests is None:
        _fail(f"unsupported explicit schema dispatch for auto identities v{schema_version}")
    specifications = (
        ("m5-controller", "TOOL", "v1", Path(__file__).resolve(), "text/x-python", None),
        ("m5-immutable-helper", "TOOL", "v1", Path(immutable.__file__).resolve(), "text/x-python", None),
        ("m5-run-schema", "SCHEMA", f"v{schema_version}", schema_paths["M5_RUN"], "application/schema+json", schema_digests["M5_RUN"]),
        ("m5-result-schema", "SCHEMA", f"v{schema_version}", schema_paths["M5_RESULT"], "application/schema+json", schema_digests["M5_RESULT"]),
        ("m5-done-schema", "SCHEMA", f"v{schema_version}", schema_paths["M5_DONE"], "application/schema+json", schema_digests["M5_DONE"]),
        ("m5-campaign-protocol", "PROTOCOL", "v2.1", CAMPAIGN_PROTOCOL_PATH, "application/json", PINNED_AUTHORITY_SHA256["campaign_protocol"]),
        ("m5-canonical-matrix", "ROSTER", "v1", CANONICAL_MATRIX_PATH, "application/json", PINNED_AUTHORITY_SHA256["canonical_matrix"]),
        ("m5-canonical-roster", "ROSTER", "v1", CANONICAL_ROSTER_PATH, "application/json", PINNED_AUTHORITY_SHA256["canonical_roster"]),
        ("m5-quantity-authority-schema", "SCHEMA", "v2", AUTHORITY_SCHEMA_PATH_V2, "application/schema+json", AUTHORITY_SCHEMA_SHA256_V2),
        ("m5-ancestry-audit-tool", "TOOL", "v1", AUTHORITY_ANCESTRY_AUDIT_TOOL_PATH_V1, "text/x-python", AUTHORITY_ANCESTRY_AUDIT_TOOL_SHA256_V1),
        ("m5-ancestry-audit-schema", "SCHEMA", "v1", AUTHORITY_ANCESTRY_AUDIT_SCHEMA_PATH_V1, "application/schema+json", AUTHORITY_ANCESTRY_AUDIT_SCHEMA_SHA256_V1),
        ("m5-source-surface-registry", "AUTHORITY", "v1", M5_SOURCE_REGISTRY_PATH_V1, "application/json", M5_SOURCE_REGISTRY_SHA256_V1),
        ("m5-source-surface-registry-schema", "SCHEMA", "v1", M5_SOURCE_REGISTRY_SCHEMA_PATH_V1, "application/schema+json", M5_SOURCE_REGISTRY_SCHEMA_SHA256_V1),
        ("m5-trust-anchor-registry", "AUTHORITY", "v1", TRUST_ANCHOR_REGISTRY_PATH_V1, TRUST_ANCHOR_REGISTRY_MEDIA_TYPE_V1, TRUST_ANCHOR_REGISTRY_SHA256_V1),
        ("m5-trust-anchor-registry-schema", "SCHEMA", "v1", TRUST_ANCHOR_SCHEMA_PATH_V1, "application/schema+json", TRUST_ANCHOR_SCHEMA_SHA256_V1),
        ("m5-control-emitter", "TOOL", "v1", CONTROL_EMITTER_PATH, "text/x-python", PINNED_AUTHORITY_SHA256["control_emitter"]),
        ("m5-control-extractor", "TOOL", "v1", CONTROL_EXTRACTOR_PATH, "text/x-python", PINNED_AUTHORITY_SHA256["control_extractor"]),
        ("m5-control-join-tool", "TOOL", "v1", CONTROL_JOIN_TOOL_PATH, "text/x-python", PINNED_AUTHORITY_SHA256["control_join_tool"]),
        ("m5-control-join-schema", "SCHEMA", "v1", CONTROL_JOIN_SCHEMA_PATH, "application/schema+json", PINNED_AUTHORITY_SHA256["control_join_schema"]),
        ("m5-pchase-source", "SOURCE", "v1", PCHASE_SOURCE_PATH, "text/x-cuda", PINNED_AUTHORITY_SHA256["pchase_source"]),
        ("m5-pchase-v1-card", "PROTOCOL", "historical-v1", PCHASE_V1_CARD_PATH, "application/json", PINNED_AUTHORITY_SHA256["pchase_v1_card"]),
    )
    for identity_id, kind, version, path, media_type, expected_digest in specifications:
        identity, file = _artifact_from_path(identity_id, kind, version, path, media_type)
        if expected_digest is not None and identity["artifact"]["sha256"] != expected_digest:
            _fail(f"controller-pinned authority/tool bytes changed for {identity_id}")
        identities.append(identity)
        _merge_material(material, file)

    quantity_rows = list(quantities)
    for quantity in quantity_rows:
        for contract in quantity["readout_contracts"]:
            derivation = {
                "schema_version": 1,
                "artifact_kind": "M5_READOUT_DERIVATION_CONTRACT",
                "quantity_id": quantity["quantity_id"],
                "readout_id": contract["readout_id"],
                "method_class": contract["method_class"],
                "instrument": contract["instrument"],
                "unit": quantity["unit"],
                "operation": contract["derivation_operation"],
                "sample_operation": contract["sample_operation"],
                "minimum_sample_count": 3,
                "ncu_inputs_forbidden": True,
                "simulator_inputs_forbidden": True,
            }
            identity, file = _artifact_from_bytes(
                contract["derivation_identity_id"],
                "DERIVATION",
                "v1",
                immutable.pretty_json_bytes(derivation),
                "application/json",
            )
            identities.append(identity)
            _merge_material(material, file)

    status_identity, status_file = _artifact_from_bytes(
        "m5-source-status",
        "SOURCE_STATUS",
        "porcelain-v1-z",
        b"",
        "application/vnd.git.status-porcelain-v1-z",
    )
    identities.append(status_identity)
    _merge_material(material, status_file)
    return identities, material


def _reauthorize_run_source_and_controller(run: Mapping[str, Any]) -> None:
    commit, status = _checked_observation(_observe_git_source, "analysis source reauthorization")
    if commit != run["source_state"]["source_commit"] or status != b"":
        _fail("live Git source differs from the clean RUN source authority")
    current_identities, _material = _auto_identities(
        run["quantities"], schema_version=run["schema_version"]
    )
    run_identities = _identity_map(run)
    for current in current_identities:
        retained = run_identities.get(current["identity_id"])
        if retained != current:
            _fail(f"live controller authority changed since RUN: {current['identity_id']}")


def _canonical_roster_component(
    matrix_payload: bytes, roster_payload: bytes
) -> dict[str, Any]:
    try:
        matrix = immutable.strict_json_loads(matrix_payload, label="canonical M5 matrix")
        roster = immutable.strict_json_loads(roster_payload, label="canonical M5 roster")
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    matrix = _exact_object(matrix, "canonical M5 matrix")
    roster = _exact_object(roster, "canonical M5 roster")
    if (
        matrix.get("schema_version") != 1
        or matrix.get("matrix_id") != "sm86_reference_capture_matrix_v1"
        or matrix.get("arch") != "sm_86"
        or matrix.get("sku") != "rtx_3090_ti"
        or matrix.get("counts", {}).get("universe") != 344
        or matrix.get("counts", {}).get("runnable") != 122
        or matrix.get("counts", {}).get("excluded") != 222
        or matrix.get("authority")
        != {
            "path": "data/local_labels/campaign/capture_roster_authority_sm86.json",
            "roster_id": "sm86_capture_roster_authority_v1",
            "sha256": PINNED_AUTHORITY_SHA256["canonical_roster"],
        }
    ):
        _fail("canonical capture matrix identity/count authority is not exact")
    if (
        roster.get("schema_version") != 1
        or roster.get("roster_id") != "sm86_capture_roster_authority_v1"
        or roster.get("arch") != "sm_86"
        or roster.get("sku") != "rtx_3090_ti"
        or roster.get("accounting_identity_holds") is not True
        or roster.get("counts", {}).get("universe") != 344
        or roster.get("counts", {}).get("runnable") != 122
        or roster.get("counts", {}).get("excluded") != 222
    ):
        _fail("canonical roster authority identity/count closure is not exact")
    roster_cells: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in _exact_array(roster.get("cells"), "canonical roster cells"):
        cell = _exact_object(raw, "canonical roster cell")
        key = (
            _exact_string(cell.get("roster_source"), "roster source"),
            _exact_string(cell.get("slug"), "roster slug"),
        )
        if key in roster_cells:
            _fail(f"canonical roster has duplicate cell {key}")
        roster_cells[key] = cell
    if len(roster_cells) != 344:
        _fail("canonical roster does not contain exactly 344 unique cells")

    members: list[dict[str, Any]] = []
    matrix_keys: set[tuple[str, str]] = set()
    for raw in _exact_array(matrix.get("cells"), "canonical matrix cells"):
        cell = _exact_object(raw, "canonical matrix cell")
        identity = _exact_object(cell.get("identity"), "canonical matrix identity")
        key = (
            _exact_string(identity.get("roster_source"), "matrix roster source"),
            _exact_string(identity.get("slug"), "matrix slug"),
        )
        if key in matrix_keys:
            _fail(f"canonical matrix has duplicate cell {key}")
        matrix_keys.add(key)
        authority_cell = roster_cells.get(key)
        if authority_cell is None:
            _fail(f"canonical matrix cell {key} is absent from roster authority")
        authority_sha = immutable.sha256_bytes(
            immutable.canonical_json_bytes(authority_cell)
        )
        if authority_sha != cell.get("authority_cell_sha256"):
            _fail(f"canonical matrix authority-cell hash mismatch for {key}")
        runnable = cell.get("disposition") == "runnable"
        if runnable != (authority_cell.get("resolution_status") == "runnable"):
            _fail(f"canonical matrix/roster disposition mismatch for {key}")
        if not runnable:
            continue
        binary_sha = _exact_string(
            authority_cell.get("binary_sha256"), "canonical runnable binary sha256"
        )
        if re.fullmatch(r"[0-9a-f]{64}", binary_sha) is None:
            _fail(f"canonical runnable binary hash is invalid for {key}")
        control = _exact_object(cell.get("control_bits"), "canonical control contract")
        cell_id = _canonical_safe_id("cell", key[0], key[1])
        members.append(
            {
                "cell_id": cell_id,
                "roster_source": key[0],
                "slug": key[1],
                "shape_point": _exact_string(
                    authority_cell.get("shape_point"), "canonical runnable shape point"
                ),
                "authority_cell_sha256": authority_sha,
                "binary_sha256": binary_sha,
                "control_contract_sha256": immutable.sha256_bytes(
                    immutable.canonical_json_bytes(control)
                ),
                "control_policy": _exact_string(
                    control.get("policy"), "canonical control policy"
                ),
                "expected_sidecar_id": f"control.{cell_id}",
            }
        )
    if matrix_keys != set(roster_cells) or len(members) != 122:
        _fail("canonical matrix/roster full-universe or runnable closure failed")
    return {
        "component_id": "canonical_runnable_roster",
        "matrix_identity_id": "m5-canonical-matrix",
        "roster_identity_id": "m5-canonical-roster",
        "matrix_id": "sm86_reference_capture_matrix_v1",
        "roster_id": "sm86_capture_roster_authority_v1",
        "universe_count": 344,
        "runnable_count": 122,
        "excluded_count": 222,
        "members": members,
    }


def _validate_run_semantics(
    run: Mapping[str, Any],
    schema: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, Any]], bytes],
) -> None:
    _verify_sealed_model(run, "M5_RUN", schema)
    source = run["source_state"]
    if len({source["source_commit"], source["pre_plan_commit"], source["post_plan_commit"]}) != 1:
        _fail("RUN source commit changed across the clean/stable observation")
    if any(
        source[field] is not True
        for field in ("working_tree_clean", "index_clean", "untracked_clean", "stable")
    ):
        _fail("RUN source state is not the exact derived clean/stable state")
    if run["target"]["architecture"] != "sm_86" or run["target"]["chip"] != "GA102" or run["target"]["sku"] != "RTX 3090 Ti":
        _fail("RUN target is not the exact sm_86 GA102 RTX 3090 Ti campaign target")
    operating = run["operating_state"]
    relationship = operating["core_memory_relationship"]
    if (
        type(operating["sm_clock_mhz"]) is not int
        or operating["sm_clock_mhz"] != 1350
        or type(operating["memory_clock_mhz"]) is not int
        or operating["memory_clock_mhz"] != 10501
        or operating["persistence_mode"] is not True
        or operating["clock_locked"] is not True
        or operating["cache_control"] != "all"
        or operating["driver_version"] != "610.43.02"
        or relationship
        != {
            "kind": "LOCKED_FIXED_PAIR",
            "core_mhz": 1350,
            "memory_mhz": 10501,
            "reduced_core": 1350,
            "reduced_memory": 10501,
        }
    ):
        _fail("RUN operating state is not the exact pinned 1350:10501 campaign point")
    identities = _identity_map(run)
    observed_kinds = {identity["identity_kind"] for identity in identities.values()}
    if missing := _REQUIRED_IDENTITY_KINDS - observed_kinds:
        _fail(f"RUN identity coverage is incomplete: {sorted(missing)}")
    status_identity_ids = {
        identity_id
        for identity_id, identity in identities.items()
        if identity["identity_kind"] == "SOURCE_STATUS"
    }
    if status_identity_ids != {"m5-source-status"}:
        _fail("RUN must contain exactly one dedicated SOURCE_STATUS identity")
    for identity_id, identity in identities.items():
        if identity["identity_kind"] != "SOURCE_STATUS" and identity["artifact"]["size_bytes"] < 1:
            _fail(f"RUN identity {identity_id} has an empty retained artifact")

    status_id = source["status_identity_id"]
    _require_identity_kind(identities, status_id, "SOURCE_STATUS", "source status")
    status_artifact = identities[status_id]["artifact"]
    if (
        status_artifact["sha256"] != _EMPTY_SHA256
        or status_artifact["size_bytes"] != 0
        or status_artifact["media_type"] != "application/vnd.git.status-porcelain-v1-z"
        or artifact_loader(status_artifact) != b""
    ):
        _fail("SOURCE_STATUS must retain the exact empty git status --porcelain=v1 -z output")

    run_schema_version = _schema_version_for_model(run, "M5_RUN")
    schema_version_text = f"v{run_schema_version}"
    exact_auto_contracts = {
        "m5-controller": ("TOOL", "v1", "text/x-python"),
        "m5-immutable-helper": ("TOOL", "v1", "text/x-python"),
        "m5-run-schema": ("SCHEMA", schema_version_text, "application/schema+json"),
        "m5-result-schema": ("SCHEMA", schema_version_text, "application/schema+json"),
        "m5-done-schema": ("SCHEMA", schema_version_text, "application/schema+json"),
        "m5-campaign-protocol": ("PROTOCOL", "v2.1", "application/json"),
        "m5-canonical-matrix": ("ROSTER", "v1", "application/json"),
        "m5-canonical-roster": ("ROSTER", "v1", "application/json"),
        "m5-quantity-authority-schema": ("SCHEMA", "v2", "application/schema+json"),
        "m5-ancestry-audit-tool": ("TOOL", "v1", "text/x-python"),
        "m5-ancestry-audit-schema": ("SCHEMA", "v1", "application/schema+json"),
        "m5-source-surface-registry": ("AUTHORITY", "v1", "application/json"),
        "m5-source-surface-registry-schema": (
            "SCHEMA",
            "v1",
            "application/schema+json",
        ),
        "m5-trust-anchor-registry": (
            "AUTHORITY",
            "v1",
            TRUST_ANCHOR_REGISTRY_MEDIA_TYPE_V1,
        ),
        "m5-trust-anchor-registry-schema": (
            "SCHEMA",
            "v1",
            "application/schema+json",
        ),
        "m5-control-emitter": ("TOOL", "v1", "text/x-python"),
        "m5-control-extractor": ("TOOL", "v1", "text/x-python"),
        "m5-control-join-tool": ("TOOL", "v1", "text/x-python"),
        "m5-control-join-schema": ("SCHEMA", "v1", "application/schema+json"),
        "m5-pchase-source": ("SOURCE", "v1", "text/x-cuda"),
        "m5-pchase-v1-card": ("PROTOCOL", "historical-v1", "application/json"),
        "m5-source-status": (
            "SOURCE_STATUS",
            "porcelain-v1-z",
            "application/vnd.git.status-porcelain-v1-z",
        ),
    }
    for identity_id, (kind, version, media_type) in exact_auto_contracts.items():
        _require_identity_kind(identities, identity_id, kind, identity_id)
        identity = identities[identity_id]
        if (
            identity["version"] != version
            or identity["artifact"]["artifact_id"] != f"{identity_id}-artifact"
            or identity["artifact"]["media_type"] != media_type
        ):
            _fail(f"RUN auto-bound identity contract is not exact for {identity_id}")
    pinned_identity_digests = {
        "m5-campaign-protocol": PINNED_AUTHORITY_SHA256["campaign_protocol"],
        "m5-canonical-matrix": PINNED_AUTHORITY_SHA256["canonical_matrix"],
        "m5-canonical-roster": PINNED_AUTHORITY_SHA256["canonical_roster"],
        "m5-quantity-authority-schema": AUTHORITY_SCHEMA_SHA256_V2,
        "m5-ancestry-audit-tool": AUTHORITY_ANCESTRY_AUDIT_TOOL_SHA256_V1,
        "m5-ancestry-audit-schema": AUTHORITY_ANCESTRY_AUDIT_SCHEMA_SHA256_V1,
        "m5-source-surface-registry": M5_SOURCE_REGISTRY_SHA256_V1,
        "m5-source-surface-registry-schema": M5_SOURCE_REGISTRY_SCHEMA_SHA256_V1,
        "m5-trust-anchor-registry": TRUST_ANCHOR_REGISTRY_SHA256_V1,
        "m5-trust-anchor-registry-schema": TRUST_ANCHOR_SCHEMA_SHA256_V1,
        "m5-control-emitter": PINNED_AUTHORITY_SHA256["control_emitter"],
        "m5-control-extractor": PINNED_AUTHORITY_SHA256["control_extractor"],
        "m5-control-join-tool": PINNED_AUTHORITY_SHA256["control_join_tool"],
        "m5-control-join-schema": PINNED_AUTHORITY_SHA256["control_join_schema"],
        "m5-pchase-source": PINNED_AUTHORITY_SHA256["pchase_source"],
        "m5-pchase-v1-card": PINNED_AUTHORITY_SHA256["pchase_v1_card"],
    }
    for identity_id, digest in pinned_identity_digests.items():
        if identities[identity_id]["artifact"]["sha256"] != digest:
            _fail(f"RUN changed controller-pinned identity bytes for {identity_id}")
    if run["tool_bindings"] != {
        "controller_identity_id": "m5-controller",
        "immutable_helper_identity_id": "m5-immutable-helper",
    }:
        _fail("RUN tool bindings are not exact")
    if run["schema_bindings"] != {
        "run_identity_id": "m5-run-schema",
        "result_identity_id": "m5-result-schema",
        "done_identity_id": "m5-done-schema",
    }:
        _fail("RUN schema bindings are not exact")
    authority_identity = identities.get("m5-analysis-authority")
    quantity_authority_identity = identities.get(QUANTITY_AUTHORITY_ID_V2)
    if (
        authority_identity is None
        or authority_identity["identity_kind"] != "AUTHORITY"
        or authority_identity["version"] != "ed25519-v1"
        or authority_identity["artifact"]["media_type"]
        != "application/vnd.m5.ed25519-public-key"
    ):
        _fail("RUN analysis authenticity authority is absent or malformed")
    authority_key = artifact_loader(authority_identity["artifact"])
    if len(authority_key) != 32:
        _fail("RUN analysis authenticity authority is not a 32-byte Ed25519 key")
    _validate_ed25519_public_key(authority_key, "RUN analysis authority public key")
    if (
        quantity_authority_identity is None
        or quantity_authority_identity["identity_kind"] != "AUTHORITY"
        or quantity_authority_identity["version"] != "v2"
        or quantity_authority_identity["artifact"]["media_type"]
        != QUANTITY_AUTHORITY_MEDIA_TYPE_V2
    ):
        _fail("M5_RULE1_PREREGISTRATION_VERSION: RUN lacks external v2 authority")
    provenance = _exact_object(run["provenance"], "RUN provenance")
    if provenance["authority_artifact"] != quantity_authority_identity["artifact"]:
        _fail("RUN provenance authority artifact is not identity-bound")
    ancestry_identity = identities.get(AUTHORITY_ANCESTRY_ID_V2)
    ancestry_audit_identity = identities.get(AUTHORITY_ANCESTRY_AUDIT_ID_V1)
    operator_identity = identities.get(AUTHORITY_OPERATOR_APPROVAL_ID_V2)
    if ancestry_identity is None or operator_identity is None:
        _fail("M5_RULE1_PREREGISTRATION_ANCESTRY: RUN witness identities are incomplete")
    if (
        ancestry_audit_identity is None
        or ancestry_audit_identity["identity_kind"] != "AUDIT"
        or ancestry_audit_identity["version"] != "v1"
        or ancestry_audit_identity["artifact"]["media_type"]
        != AUTHORITY_ANCESTRY_AUDIT_MEDIA_TYPE_V1
    ):
        _fail("M5_RULE1_PREREGISTRATION_ANCESTRY: RUN computed audit is absent")
    ancestry = _authority_witness(
        artifact_loader(ancestry_identity["artifact"]),
        ancestry_identity["artifact"],
        AUTHORITY_ANCESTRY_ID_V2,
    )
    operator_approval = _authority_witness(
        artifact_loader(operator_identity["artifact"]),
        operator_identity["artifact"],
        AUTHORITY_OPERATOR_APPROVAL_ID_V2,
    )
    authority_raw = artifact_loader(quantity_authority_identity["artifact"])
    try:
        quantity_authority = immutable.strict_json_loads(
            authority_raw, label="M5 v2 quantity authority"
        )
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    quantity_authority = _exact_object(quantity_authority, "M5 v2 quantity authority")
    authority_provenance = _exact_object(
        quantity_authority.get("provenance"), "M5 v2 authority provenance"
    )
    review_references: list[Mapping[str, Any]] = []
    review_witnesses: list[Mapping[str, Any]] = []
    review_identity_ids: list[str] = []
    for review_id in _exact_array(
        authority_provenance.get("review_identity_ids"),
        "M5 v2 authority review identities",
    ):
        normalized_review_id = _safe_id(review_id, "review identity_id")
        review_identity = identities.get(normalized_review_id)
        if review_identity is None or review_identity["identity_kind"] != "AUTHORITY":
            _fail("M5_RULE1_INDEPENDENT_REVIEW_MISSING: RUN review identity is absent")
        review_identity_ids.append(normalized_review_id)
        review_references.append(review_identity["artifact"])
        review_witnesses.append(
            _authority_witness(
                artifact_loader(review_identity["artifact"]),
                review_identity["artifact"],
                review_identity["identity_id"],
            )
        )
    _validate_v2_quantity_authority(
        quantity_authority,
        authority_raw,
        authority_reference=quantity_authority_identity["artifact"],
        ancestry_reference=ancestry_identity["artifact"],
        review_references=review_references,
        operator_reference=operator_identity["artifact"],
        ancestry=ancestry,
        review_witnesses=review_witnesses,
        operator_approval=operator_approval,
        trust_anchor=authority_key,
        schema_payload=artifact_loader(
            identities["m5-quantity-authority-schema"]["artifact"]
        ),
    )
    design_inputs = _authority_design_inputs(
        quantity_authority, identities, artifact_loader
    )
    _audit, expected_audit_payload = _build_ancestry_audit(
        authority=quantity_authority,
        authority_raw=authority_raw,
        design_inputs=design_inputs,
        source_registry_payload=artifact_loader(
            identities["m5-source-surface-registry"]["artifact"]
        ),
        source_registry_schema_payload=artifact_loader(
            identities["m5-source-surface-registry-schema"]["artifact"]
        ),
        audit_tool_payload=artifact_loader(
            identities["m5-ancestry-audit-tool"]["artifact"]
        ),
        schema_payload=artifact_loader(
            identities["m5-ancestry-audit-schema"]["artifact"]
        ),
    )
    retained_audit_payload = artifact_loader(ancestry_audit_identity["artifact"])
    if retained_audit_payload != expected_audit_payload:
        _fail(
            "M5_RULE1_PREREGISTRATION_ANCESTRY: retained audit differs from "
            "controller recomputation"
        )
    _require_admitted_v2_authority(
        public_key=authority_key,
        authority_reference=quantity_authority_identity["artifact"],
        ancestry_reference=ancestry_identity["artifact"],
        ancestry_audit_reference=ancestry_audit_identity["artifact"],
        audit_policy_sha256=_audit["audit_policy_sha256"],
        design_input_references={
            identity_id: reference
            for identity_id, (reference, _payload) in design_inputs.items()
        },
        review_identity_ids=review_identity_ids,
        review_references=review_references,
        operator_reference=operator_identity["artifact"],
        registry_payload=artifact_loader(
            identities["m5-trust-anchor-registry"]["artifact"]
        ),
        registry_schema_payload=artifact_loader(
            identities["m5-trust-anchor-registry-schema"]["artifact"]
        ),
    )
    if (
        run["provenance"]["ancestry_artifact"] != ancestry_identity["artifact"]
        or run["provenance"]["ancestry_audit_artifact"]
        != ancestry_audit_identity["artifact"]
        or run["provenance"]["design_input_artifacts"]
        != [design_inputs[identity_id][0] for identity_id in sorted(design_inputs)]
        or run["provenance"]["review_artifacts"] != review_references
    ):
        _fail("RUN provenance witness artifacts are not identity-bound")
    if run["provenance"]["target_free"] is not True or run["provenance"]["operator_approved"] is not True:
        _fail("RUN provenance does not attest target-free operator approval")
    _campaign_protocol_model(artifact_loader(identities["m5-campaign-protocol"]["artifact"]))
    if run["protocol"] != {
        "protocol_id": "campaign_protocol_pins",
        "protocol_version": "v2.1",
        "protocol_identity_id": "m5-campaign-protocol",
        "clock_control": "none",
        "cache_control": "all",
        "ncu_role": "CROSS_CHECK_ONLY",
    }:
        _fail("RUN protocol is not the exact pinned v2.1 contract")
    if run["policy"] != {
        "ncu_cross_check_only": True,
        "ncu_derivation_forbidden": True,
        "ncu_selection_forbidden": True,
        "ncu_tuning_forbidden": True,
        "ncu_transcription_forbidden": True,
        "simulator_fit_forbidden": True,
        "architecture_card_write_forbidden": True,
        "publication_mode": "ADDITIVE_EVIDENCE_ONLY",
    }:
        _fail("RUN policy is not the exact Rule-1/additive-only controller policy")

    pin_ids: set[str] = set()
    for index, pin in enumerate(run["pins"]):
        if pin["pin_id"] in pin_ids:
            _fail("RUN pin_id values are not unique")
        pin_ids.add(pin["pin_id"])
        identity = identities.get(pin["authority_identity_id"])
        if identity is None or identity["identity_kind"] != pin["authority_kind"]:
            _fail(f"pins[{index}] authority_kind does not exactly match its identity")

    required_auto_pins = {
        "campaign-protocol-version": ("m5-campaign-protocol", "PROTOCOL", "v2.1"),
        "canonical-runnable-roster": ("m5-canonical-matrix", "ROSTER", "344/122/222-v1"),
        "m5-quantity-authority": (
            QUANTITY_AUTHORITY_ID_V2,
            "AUTHORITY",
            "sm86-m5-quantity-authority-v2",
        ),
        "analysis-receipt-authority": (
            "m5-analysis-authority",
            "AUTHORITY",
            authority_identity["artifact"]["sha256"],
        ),
    }
    pin_by_id = {pin["pin_id"]: pin for pin in run["pins"]}
    for pin_id, (identity_id, kind, value) in required_auto_pins.items():
        pin = pin_by_id.get(pin_id)
        if pin != {
            "pin_id": pin_id,
            "value": value,
            "authority_identity_id": identity_id,
            "authority_kind": kind,
        }:
            _fail(f"RUN required authority pin is absent or changed: {pin_id}")

    relationships: dict[str, Mapping[str, Any]] = {}
    relationship_binaries: dict[str, str] = {}
    for relationship in run["build_relationships"]:
        relationship_id = relationship["relationship_id"]
        if relationship_id in relationships:
            _fail(f"duplicate build relationship: {relationship_id}")
        relationships[relationship_id] = relationship
        for source_id in relationship["source_identity_ids"]:
            _require_identity_kind(identities, source_id, "SOURCE", relationship_id)
        _require_identity_kind(
            identities, relationship["build_identity_id"], "BUILD", relationship_id
        )
        _require_identity_kind(
            identities, relationship["binary_identity_id"], "BINARY", relationship_id
        )
        relationship_binaries[relationship_id] = relationship["binary_identity_id"]

    execution_order = tuple(quantity_authority["execution_order"])
    scientific_rank = tuple(quantity_authority["scientific_rank"])
    if tuple(run["family_execution_order"]) != execution_order:
        _fail("RUN family execution order differs from the external v2 authority")
    if tuple(run["family_scientific_rank"]) != scientific_rank:
        _fail("RUN family scientific rank differs from the external v2 authority")
    families = run["families"]
    if tuple(row["family_id"] for row in families) != execution_order:
        _fail("RUN family records are not in exact authority execution order")
    rank = {family: index + 1 for index, family in enumerate(scientific_rank)}
    execution_position = {
        family: index for index, family in enumerate(execution_order)
    }
    family_binary_ids: set[str] = set()
    referenced_relationship_ids: set[str] = set()
    for execution_index, family in enumerate(families, start=1):
        family_id = family["family_id"]
        if family["execution_index"] != execution_index or family["scientific_rank"] != rank[family_id]:
            _fail(f"family {family_id} order/rank mismatch")
        for harness_id in family["harness_identity_ids"]:
            _require_identity_kind(identities, harness_id, "HARNESS", family_id)
        for relationship_id in family["build_relationship_ids"]:
            if relationship_id not in relationships:
                _fail(f"family {family_id} references absent build relationship")
            referenced_relationship_ids.add(relationship_id)
            family_binary_ids.add(relationship_binaries[relationship_id])

    quantities = run["quantities"]
    if quantities != quantity_authority["quantities"]:
        _fail("RUN quantities differ from the externally retained v2 authority")
    quantity_ids: set[str] = set()
    conditioned_bindings: set[bytes] = set()
    family_quantities: dict[str, int] = defaultdict(int)
    probe_execution_positions: list[int] = []
    pchase_ids: list[str] = []
    for quantity in quantities:
        quantity_id = quantity["quantity_id"]
        if quantity_id in quantity_ids:
            _fail("RUN quantity_id values are not unique")
        quantity_ids.add(quantity_id)
        condition_key = immutable.canonical_json_bytes(
            [quantity["consumer_binding"], quantity["conditioning"]]
        )
        if condition_key in conditioned_bindings:
            _fail("RUN consumer binding plus conditioning must be unique")
        conditioned_bindings.add(condition_key)
        producer = quantity["producer"]
        upstream_family_ids = quantity["upstream_family_ids"]
        if producer["kind"] == "PROBE_FAMILY":
            family_id = producer["family_id"]
            family_quantities[family_id] += 1
            producer_position = execution_position[family_id]
            probe_execution_positions.append(producer_position)
            if any(
                upstream_id not in execution_position
                for upstream_id in upstream_family_ids
            ):
                _fail(f"quantity {quantity_id} has an absent upstream family")
            if upstream_family_ids != sorted(
                upstream_family_ids, key=execution_position.__getitem__
            ):
                _fail(
                    f"quantity {quantity_id} upstream families are not in authority execution order"
                )
            if any(
                execution_position[upstream_id] >= producer_position
                for upstream_id in upstream_family_ids
            ):
                _fail(
                    f"quantity {quantity_id} has a self or non-prior upstream family"
                )
        else:
            if upstream_family_ids:
                _fail(f"pchase quantity {quantity_id} cannot declare probe-family dependencies")
            pchase_ids.append(quantity_id)

        contracts = quantity["readout_contracts"]
        if len(contracts) > 16:
            _fail("quantity readout contract exceeds the bounded maximum of 16")
        readout_ids = [contract["readout_id"] for contract in contracts]
        methods = [contract["method_class"] for contract in contracts]
        derivations = [contract["derivation_identity_id"] for contract in contracts]
        if len(readout_ids) != len(set(readout_ids)):
            _fail(f"quantity {quantity_id} has duplicate planned readout IDs")
        if len(set(methods)) < 2:
            _fail(f"quantity {quantity_id} lacks two independent registered methods")
        if len(derivations) != len(set(derivations)):
            _fail(f"quantity {quantity_id} reuses a derivation identity")
        derivation_hashes: list[str] = []
        for contract in contracts:
            derivation_id = contract["derivation_identity_id"]
            _require_identity_kind(identities, derivation_id, "DERIVATION", quantity_id)
            derivation_hashes.append(identities[derivation_id]["artifact"]["sha256"])
            expected_derivation = {
                "schema_version": 1,
                "artifact_kind": "M5_READOUT_DERIVATION_CONTRACT",
                "quantity_id": quantity_id,
                "readout_id": contract["readout_id"],
                "method_class": contract["method_class"],
                "instrument": contract["instrument"],
                "unit": quantity["unit"],
                "operation": contract["derivation_operation"],
                "sample_operation": contract["sample_operation"],
                "minimum_sample_count": 3,
                "ncu_inputs_forbidden": True,
                "simulator_inputs_forbidden": True,
            }
            payload = artifact_loader(identities[derivation_id]["artifact"])
            try:
                observed_derivation = immutable.strict_json_loads(
                    payload, label=f"derivation contract {derivation_id}"
                )
            except immutable.ImmutableArtifactError as exc:
                _fail(str(exc))
            if observed_derivation != expected_derivation:
                _fail(f"quantity {quantity_id} derivation contract is not exact")
        if len(derivation_hashes) != len(set(derivation_hashes)):
            _fail(f"quantity {quantity_id} derivation identities are not byte-distinct")
    if probe_execution_positions != sorted(probe_execution_positions):
        _fail("RUN probe quantities are not grouped in authority execution order")
    if set(family_quantities) != set(execution_order):
        _fail("every authority family must own a consumer-bound quantity")
    if not pchase_ids:
        _fail("pchase must remain an additional required component")

    pchase = run["components"]["pchase"]
    if pchase["quantity_ids"] != pchase_ids:
        _fail("RUN pchase quantity inventory mismatch")
    pchase_binary_ids: set[str] = set()
    for harness_id in pchase["harness_identity_ids"]:
        _require_identity_kind(identities, harness_id, "HARNESS", "pchase")
    for relationship_id in pchase["build_relationship_ids"]:
        if relationship_id not in relationships:
            _fail("pchase references absent build relationship")
        referenced_relationship_ids.add(relationship_id)
        pchase_binary_ids.add(relationship_binaries[relationship_id])

    if not any(
        "m5-pchase-source" in relationships[relationship_id]["source_identity_ids"]
        for relationship_id in pchase["build_relationship_ids"]
    ):
        _fail("RUN pchase build closure does not bind the pinned pchase source")

    if referenced_relationship_ids != set(relationships):
        _fail("RUN contains an orphan source-to-build-to-binary relationship")

    canonical = run["components"]["canonical_runnable_roster"]
    expected_canonical = _canonical_roster_component(
        artifact_loader(identities["m5-canonical-matrix"]["artifact"]),
        artifact_loader(identities["m5-canonical-roster"]["artifact"]),
    )
    if canonical != expected_canonical:
        _fail("RUN canonical 344/122/222 roster component is not byte-authority-derived")

    control = run["components"]["roster_control_sidecars"]
    expected_control_keys = {
        "component_id",
        "required",
        "canonical_roster_component_id",
        "emitter_identity_id",
        "extractor_identity_id",
        "pc_join_tool_identity_id",
        "pc_join_schema_identity_id",
        "parser_contract",
        "parser_contract_sha256",
        "expected_inventory_id",
        "expected_canonical_pc_join_witness_id",
        "expected_calibration_pc_join_witness_id",
        "expected_canonical_cell_count",
        "expected_calibration_binary_count",
        "calibration_members",
    }
    if set(control) != expected_control_keys:
        _fail("RUN control component is not closed")
    if (
        control["component_id"] != "roster_control_sidecars"
        or control["required"] is not True
        or control["canonical_roster_component_id"] != "canonical_runnable_roster"
        or control["emitter_identity_id"] != "m5-control-emitter"
        or control["extractor_identity_id"] != "m5-control-extractor"
        or control["pc_join_tool_identity_id"] != "m5-control-join-tool"
        or control["pc_join_schema_identity_id"] != "m5-control-join-schema"
        or control["parser_contract"] != CONTROL_PARSER_CONTRACT
        or control["parser_contract_sha256"]
        != immutable.sha256_bytes(immutable.canonical_json_bytes(CONTROL_PARSER_CONTRACT))
        or control["expected_inventory_id"] != "control-inventory"
        or control["expected_canonical_pc_join_witness_id"] != "canonical-pc-join-witness"
        or control["expected_calibration_pc_join_witness_id"] != "calibration-pc-join-witness"
        or control["expected_canonical_cell_count"] != 122
    ):
        _fail("RUN roster-control producer/parser/canonical authority binding is not exact")
    for identity_id, kind in (
        ("m5-control-emitter", "TOOL"),
        ("m5-control-extractor", "TOOL"),
        ("m5-control-join-tool", "TOOL"),
        ("m5-control-join-schema", "SCHEMA"),
    ):
        _require_identity_kind(identities, identity_id, kind, "control tooling")

    members = control["calibration_members"]
    if control["expected_calibration_binary_count"] != len(members):
        _fail("RUN calibration control-sidecar count mismatch")
    member_binary_ids = [member["binary_identity_id"] for member in members]
    member_ids = [member["cell_id"] for member in members]
    sidecar_ids = [member["expected_sidecar_id"] for member in members]
    if (
        len(member_binary_ids) != len(set(member_binary_ids))
        or len(member_ids) != len(set(member_ids))
        or len(sidecar_ids) != len(set(sidecar_ids))
    ):
        _fail("RUN calibration control roster contains duplicate member identities")
    expected_binary_ids = family_binary_ids | pchase_binary_ids
    if set(member_binary_ids) != expected_binary_ids:
        _fail("RUN calibration controls are not the exact family+pchase binary union")
    for member in members:
        if set(member) != {
            "cell_id",
            "binary_id",
            "binary_identity_id",
            "binary_sha256",
            "slug",
            "shape_point",
            "expected_sidecar_id",
        }:
            _fail("RUN calibration control member is not closed")
        _require_identity_kind(
            identities, member["binary_identity_id"], "BINARY", member["binary_id"]
        )
        if member["binary_sha256"] != identities[member["binary_identity_id"]]["artifact"]["sha256"]:
            _fail("RUN calibration control member binary SHA is stale")
        _safe_id(member["cell_id"], "calibration control cell_id")
        _safe_id(member["expected_sidecar_id"], "calibration expected sidecar ID")
        if not member["slug"] or not member["shape_point"]:
            _fail("RUN calibration control member lacks witness coordinates")

    expected_counts = {
        "family_count": len(families),
        "quantity_count": len(quantities),
        "pchase_quantity_count": len(pchase_ids),
        "canonical_runnable_cell_count": 122,
        "calibration_binary_count": len(members),
        "build_relationship_count": len(relationships),
    }
    if run["counts"] != expected_counts:
        _fail(f"RUN counts mismatch: expected {expected_counts}, observed {run['counts']}")


def _plan_request_to_run(
    request: Mapping[str, Any],
    request_path: Path,
    source_observer: Callable[[], tuple[str, bytes]],
) -> tuple[dict[str, Any], dict[str, immutable.ImmutableFile]]:
    _require_keys(
        request,
        (
            "schema_version",
            "request_kind",
            "expected_source_commit",
            "target",
            "operating_state",
            "pins",
            "identities",
            "build_relationships",
            "family_execution_order",
            "family_scientific_rank",
            "families",
            "quantities",
            "components",
            "policy",
        ),
        "plan request",
    )
    if _exact_int(request["schema_version"], "plan request schema_version") != ACTIVE_SCHEMA_VERSION:
        _fail("unsupported explicit plan request schema_version: v2 is required")
    if request["request_kind"] != "M5_PLAN_REQUEST":
        _fail("plan request_kind must be M5_PLAN_REQUEST")
    expected_commit = _exact_string(request["expected_source_commit"], "expected_source_commit")
    pre_commit, pre_status = _checked_observation(source_observer, "pre-plan")
    if expected_commit != pre_commit:
        _fail("caller expected_source_commit differs from live Git HEAD")
    if pre_status != b"":
        _fail("live Git status --porcelain=v1 -z is not exactly empty")

    caller_identities: list[dict[str, Any]] = []
    caller_material: dict[str, immutable.ImmutableFile] = {}
    caller_ids: set[str] = set()
    for index, raw_identity in enumerate(_exact_array(request["identities"], "identities")):
        label = f"plan request identities[{index}]"
        identity = _exact_object(raw_identity, label)
        _require_keys(identity, ("identity_id", "identity_kind", "version", "artifact_input"), label)
        identity_id = _safe_id(identity["identity_id"], f"{label}.identity_id")
        if identity_id == QUANTITY_AUTHORITY_ID_V1:
            _fail("M5_RULE1_PREREGISTRATION_VERSION: v1 quantity authority is quarantined")
        if identity_id == AUTHORITY_ANCESTRY_AUDIT_ID_V1:
            _fail(
                "M5_RULE1_PREREGISTRATION_ANCESTRY: caller-supplied ancestry audit is forbidden"
            )
        if identity_id in caller_ids:
            _fail(f"duplicate caller identity {identity_id}")
        caller_ids.add(identity_id)
        reference, file = _artifact_input(
            identity["artifact_input"], base=request_path.parent, label=f"{label}.artifact_input"
        )
        caller_identities.append(
            {
                "identity_id": identity_id,
                "identity_kind": identity["identity_kind"],
                "version": identity["version"],
                "artifact": reference,
            }
        )
        _merge_material(caller_material, file)

    caller_by_id = {identity["identity_id"]: identity for identity in caller_identities}
    analysis_authority = caller_by_id.get("m5-analysis-authority")
    quantity_authority_identity = caller_by_id.get(QUANTITY_AUTHORITY_ID_V2)
    if (
        analysis_authority is None
        or analysis_authority["identity_kind"] != "AUTHORITY"
        or analysis_authority["version"] != "ed25519-v1"
        or analysis_authority["artifact"]["media_type"]
        != "application/vnd.m5.ed25519-public-key"
        or analysis_authority["artifact"]["size_bytes"] != 32
    ):
        _fail("plan must bind exactly one m5-analysis-authority Ed25519 public key")
    if (
        quantity_authority_identity is None
        or quantity_authority_identity["identity_kind"] != "AUTHORITY"
        or quantity_authority_identity["version"] != "v2"
        or quantity_authority_identity["artifact"]["media_type"]
        != QUANTITY_AUTHORITY_MEDIA_TYPE_V2
    ):
        _fail("M5_RULE1_PREREGISTRATION_VERSION: plan must bind an external v2 quantity authority")
    authority_material = caller_material.get(analysis_authority["artifact"]["relative_path"])
    quantity_material = caller_material.get(
        quantity_authority_identity["artifact"]["relative_path"]
    )
    if authority_material is None or quantity_material is None:
        _fail("plan authority bytes are absent")
    analysis_public_key = _material_bytes(authority_material)
    _validate_ed25519_public_key(analysis_public_key, "plan analysis authority public key")

    def caller_artifact_payload(reference: Mapping[str, Any]) -> bytes:
        material_file = caller_material.get(reference["relative_path"])
        if material_file is None:
            _fail(f"M5 caller artifact bytes are absent: {reference['relative_path']}")
        return _material_bytes(material_file)

    def caller_witness(identity_id: str, kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
        identity = caller_by_id.get(identity_id)
        if identity is None or identity["identity_kind"] != kind:
            _fail(
                f"M5_RULE1_INDEPENDENT_REVIEW_MISSING: authority witness {identity_id} is absent or wrong-kind"
            )
        material_file = caller_material.get(identity["artifact"]["relative_path"])
        if material_file is None:
            _fail(f"M5 v2 authority witness {identity_id} bytes are absent")
        payload = _material_bytes(material_file)
        return _authority_witness(payload, identity["artifact"], identity_id), identity["artifact"]

    authority_raw = _material_bytes(quantity_material)
    try:
        authority_value = immutable.strict_json_loads(
            authority_raw, label="M5 v2 quantity authority"
        )
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    authority_value = _exact_object(authority_value, "M5 v2 quantity authority")
    ancestry, ancestry_reference = caller_witness(AUTHORITY_ANCESTRY_ID_V2, "AUTHORITY")
    review_identity_ids = _exact_array(
        _exact_object(authority_value.get("provenance"), "M5 v2 authority provenance").get(
            "review_identity_ids"
        ),
        "M5 v2 authority review identities",
    )
    review_witnesses: list[dict[str, Any]] = []
    review_references: list[dict[str, Any]] = []
    normalized_review_identity_ids: list[str] = []
    for identity_id in review_identity_ids:
        normalized_identity_id = _safe_id(
            identity_id, "M5 v2 review identity_id"
        )
        review, reference = caller_witness(normalized_identity_id, "AUTHORITY")
        normalized_review_identity_ids.append(normalized_identity_id)
        review_witnesses.append(review)
        review_references.append(reference)
    operator_approval, operator_reference = caller_witness(
        AUTHORITY_OPERATOR_APPROVAL_ID_V2, "AUTHORITY"
    )
    authority_value = _validate_v2_quantity_authority(
        authority_value,
        authority_raw,
        authority_reference=quantity_authority_identity["artifact"],
        ancestry_reference=ancestry_reference,
        review_references=review_references,
        operator_reference=operator_reference,
        ancestry=ancestry,
        review_witnesses=review_witnesses,
        operator_approval=operator_approval,
        trust_anchor=analysis_public_key,
    )
    design_inputs = _authority_design_inputs(
        authority_value, caller_by_id, caller_artifact_payload
    )
    try:
        source_registry_payload = immutable.read_regular(M5_SOURCE_REGISTRY_PATH_V1)
        source_registry_schema_payload = immutable.read_regular(
            M5_SOURCE_REGISTRY_SCHEMA_PATH_V1
        )
        audit_tool_payload = immutable.read_regular(
            AUTHORITY_ANCESTRY_AUDIT_TOOL_PATH_V1
        )
        audit_schema_payload = immutable.read_regular(
            AUTHORITY_ANCESTRY_AUDIT_SCHEMA_PATH_V1
        )
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    audit, audit_payload = _build_ancestry_audit(
        authority=authority_value,
        authority_raw=authority_raw,
        design_inputs=design_inputs,
        source_registry_payload=source_registry_payload,
        source_registry_schema_payload=source_registry_schema_payload,
        audit_tool_payload=audit_tool_payload,
        schema_payload=audit_schema_payload,
    )
    audit_identity, audit_file = _artifact_from_bytes(
        AUTHORITY_ANCESTRY_AUDIT_ID_V1,
        "AUDIT",
        "v1",
        audit_payload,
        AUTHORITY_ANCESTRY_AUDIT_MEDIA_TYPE_V1,
    )
    _require_admitted_v2_authority(
        public_key=analysis_public_key,
        authority_reference=quantity_authority_identity["artifact"],
        ancestry_reference=ancestry_reference,
        ancestry_audit_reference=audit_identity["artifact"],
        audit_policy_sha256=audit["audit_policy_sha256"],
        design_input_references={
            identity_id: reference
            for identity_id, (reference, _payload) in design_inputs.items()
        },
        review_identity_ids=normalized_review_identity_ids,
        review_references=review_references,
        operator_reference=operator_reference,
    )
    quantities = copy.deepcopy(_exact_array(request["quantities"], "quantities"))
    if quantities != authority_value["quantities"]:
        _fail("plan quantities are not the exact externally retained v2 quantity authority")
    auto_identities, material = _auto_identities(
        quantities, schema_version=ACTIVE_SCHEMA_VERSION
    )
    auto_ids = {identity["identity_id"] for identity in auto_identities}
    if auto_ids & caller_ids:
        _fail(f"caller identity collides with reserved identity {sorted(auto_ids & caller_ids)}")
    identities = list(auto_identities) + [audit_identity] + caller_identities
    _merge_material(material, audit_file)
    for file in caller_material.values():
        _merge_material(material, file)
    identity_by_id = {identity["identity_id"]: identity for identity in identities}

    pins = [
        {
            "pin_id": "campaign-protocol-version",
            "value": "v2.1",
            "authority_identity_id": "m5-campaign-protocol",
            "authority_kind": "PROTOCOL",
        },
        {
            "pin_id": "canonical-runnable-roster",
            "value": "344/122/222-v1",
            "authority_identity_id": "m5-canonical-matrix",
            "authority_kind": "ROSTER",
        },
        {
            "pin_id": "m5-quantity-authority",
            "value": "sm86-m5-quantity-authority-v2",
            "authority_identity_id": QUANTITY_AUTHORITY_ID_V2,
            "authority_kind": "AUTHORITY",
        },
        {
            "pin_id": "analysis-receipt-authority",
            "value": analysis_authority["artifact"]["sha256"],
            "authority_identity_id": "m5-analysis-authority",
            "authority_kind": "AUTHORITY",
        },
        *copy.deepcopy(_exact_array(request["pins"], "pins")),
    ]
    quantities = copy.deepcopy(_exact_array(request["quantities"], "quantities"))
    if quantities != authority_value["quantities"]:
        _fail("plan quantities are not the exact externally retained v2 quantity authority")
    request_components = copy.deepcopy(
        _exact_object(request["components"], "components")
    )
    _require_keys(
        request_components,
        ("pchase", "roster_control_sidecars"),
        "plan request components",
    )
    pchase = _exact_object(request_components["pchase"], "pchase component")
    pchase_ids = [
        quantity["quantity_id"]
        for quantity in quantities
        if quantity["producer"]["kind"] == "PCHASE"
    ]
    if pchase.get("quantity_ids") != pchase_ids:
        _fail("plan pchase component must preserve every authority pchase quantity")
    control_request = _exact_object(
        request_components["roster_control_sidecars"], "roster controls"
    )
    _require_keys(
        control_request,
        (
            "component_id",
            "required",
            "expected_inventory_id",
            "expected_binary_count",
            "members",
        ),
        "plan request roster controls",
    )
    raw_calibration_members = _exact_array(
        control_request["members"], "calibration control members"
    )
    if control_request["expected_binary_count"] != len(raw_calibration_members):
        _fail("plan request calibration control count is stale")
    calibration_members: list[dict[str, Any]] = []
    for index, raw_member in enumerate(raw_calibration_members):
        label = f"calibration control members[{index}]"
        member = _exact_object(raw_member, label)
        _require_keys(
            member,
            (
                "binary_id",
                "binary_identity_id",
                "expected_sidecar_id",
                "witness_slug",
                "shape_point",
            ),
            label,
        )
        binary_id = _safe_id(member["binary_id"], f"{label}.binary_id")
        binary_identity_id = _safe_id(
            member["binary_identity_id"], f"{label}.binary_identity_id"
        )
        binary_identity = identity_by_id.get(binary_identity_id)
        if binary_identity is None or binary_identity["identity_kind"] != "BINARY":
            _fail(f"{label} does not bind a BINARY identity")
        calibration_members.append(
            {
                "cell_id": _canonical_safe_id("calibration", binary_id),
                "binary_id": binary_id,
                "binary_identity_id": binary_identity_id,
                "binary_sha256": binary_identity["artifact"]["sha256"],
                "slug": _exact_string(member["witness_slug"], f"{label}.witness_slug"),
                "shape_point": _exact_string(member["shape_point"], f"{label}.shape_point"),
                "expected_sidecar_id": _safe_id(
                    member["expected_sidecar_id"], f"{label}.expected_sidecar_id"
                ),
            }
        )
    auto_map = {identity["identity_id"]: identity for identity in auto_identities}
    snapshot_loader = _DeclaredSnapshotLoader(material=material)
    canonical_roster = _canonical_roster_component(
        snapshot_loader(auto_map["m5-canonical-matrix"]["artifact"]),
        snapshot_loader(auto_map["m5-canonical-roster"]["artifact"]),
    )
    components = {
        "canonical_runnable_roster": canonical_roster,
        "pchase": pchase,
        "roster_control_sidecars": {
            "component_id": "roster_control_sidecars",
            "required": True,
            "canonical_roster_component_id": "canonical_runnable_roster",
            "emitter_identity_id": "m5-control-emitter",
            "extractor_identity_id": "m5-control-extractor",
            "pc_join_tool_identity_id": "m5-control-join-tool",
            "pc_join_schema_identity_id": "m5-control-join-schema",
            "parser_contract": copy.deepcopy(CONTROL_PARSER_CONTRACT),
            "parser_contract_sha256": immutable.sha256_bytes(
                immutable.canonical_json_bytes(CONTROL_PARSER_CONTRACT)
            ),
            "expected_inventory_id": "control-inventory",
            "expected_canonical_pc_join_witness_id": "canonical-pc-join-witness",
            "expected_calibration_pc_join_witness_id": "calibration-pc-join-witness",
            "expected_canonical_cell_count": 122,
            "expected_calibration_binary_count": len(calibration_members),
            "calibration_members": calibration_members,
        },
    }
    pchase_count = len(pchase_ids)
    snapshot_loader.freeze_into(material)
    relationships = copy.deepcopy(
        _exact_array(request["build_relationships"], "build_relationships")
    )

    post_commit, post_status = _checked_observation(source_observer, "post-plan")
    if (post_commit, post_status) != (pre_commit, pre_status):
        _fail("live Git source observation changed while constructing RUN")
    unsigned = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "artifact_kind": "M5_RUN",
        "source_state": {
            "source_commit": pre_commit,
            "pre_plan_commit": pre_commit,
            "post_plan_commit": post_commit,
            "working_tree_clean": True,
            "index_clean": True,
            "untracked_clean": True,
            "stable": True,
            "status_identity_id": "m5-source-status",
        },
        "target": copy.deepcopy(request["target"]),
        "operating_state": copy.deepcopy(request["operating_state"]),
        "protocol": {
            "protocol_id": "campaign_protocol_pins",
            "protocol_version": "v2.1",
            "protocol_identity_id": "m5-campaign-protocol",
            "clock_control": "none",
            "cache_control": "all",
            "ncu_role": "CROSS_CHECK_ONLY",
        },
        "pins": pins,
        "identities": identities,
        "tool_bindings": {
            "controller_identity_id": "m5-controller",
            "immutable_helper_identity_id": "m5-immutable-helper",
        },
        "schema_bindings": {
            "run_identity_id": "m5-run-schema",
            "result_identity_id": "m5-result-schema",
            "done_identity_id": "m5-done-schema",
        },
        "build_relationships": relationships,
        "family_execution_order": copy.deepcopy(request["family_execution_order"]),
        "family_scientific_rank": copy.deepcopy(request["family_scientific_rank"]),
        "families": copy.deepcopy(request["families"]),
        "quantities": quantities,
        "components": components,
        "counts": {
            "family_count": len(_exact_array(request["families"], "families")),
            "quantity_count": len(quantities),
            "pchase_quantity_count": pchase_count,
            "canonical_runnable_cell_count": 122,
            "calibration_binary_count": len(calibration_members),
            "build_relationship_count": len(relationships),
        },
        "policy": copy.deepcopy(request["policy"]),
        "provenance": {
            "authority_artifact": quantity_authority_identity["artifact"],
            "ancestry_artifact": ancestry_reference,
            "ancestry_audit_artifact": audit_identity["artifact"],
            "design_input_artifacts": [
                design_inputs[identity_id][0]
                for identity_id in sorted(design_inputs)
            ],
            "review_artifacts": review_references,
            "target_free": True,
            "operator_approved": True,
        },
    }
    run = immutable.seal_model(unsigned)
    loader = _DeclaredSnapshotLoader(material=material)
    _validate_run_semantics(run, _live_schema("M5_RUN"), loader)
    loader.freeze_into(material)
    return run, material


def _plan_with_observer(
    request_path: Path,
    output_root: Path,
    source_observer: Callable[[], tuple[str, bytes]],
) -> Path:
    request_path = Path(os.path.abspath(os.fspath(request_path)))
    _refuse_card_path(request_path, "plan request")
    root = _external_output_root(output_root)
    request, _ = _load_json_file(request_path, "M5 plan request")
    run, material = _plan_request_to_run(request, request_path, source_observer)
    run_bytes = immutable.pretty_json_bytes(run)
    material["RUN.json"] = immutable.ImmutableFile.from_bytes("RUN.json", run_bytes)

    def commit_guard(commit_root: Path, descriptor: int) -> None:
        _authorize_output_descriptor(commit_root, descriptor)
        observation = _checked_observation(source_observer, "pre-publication")
        expected = (run["source_state"]["source_commit"], b"")
        if observation != expected:
            _fail("live Git source changed before atomic RUN publication")

    try:
        return immutable.publish_directory(
            root,
            f"m5-run-{run['model_seal_sha256']}",
            material.values(),
            before_commit=commit_guard,
        )
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))


@_contract_boundary
def plan(request_path: Path, output_root: Path) -> Path:
    return _plan_with_observer(request_path, output_root, _observe_git_source)


def _bundle_path(value: Path, manifest_name: str, label: str) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(os.fspath(value)))
    _refuse_card_path(absolute, label)
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as exc:
        _fail(f"cannot inspect {label}: {exc}")
    if stat.S_ISLNK(mode):
        _fail(f"{label} path must not be a symlink")
    if stat.S_ISDIR(mode):
        return absolute, manifest_name
    if stat.S_ISREG(mode):
        return absolute.parent, absolute.name
    _fail(f"{label} path must be a directory or regular manifest")


def _run_artifacts(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for index, identity in enumerate(run["identities"]):
        _collect_artifact(references, identity["artifact"], f"RUN identity[{index}]")
    return references


def _bootstrap_identity_map(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse only the closed identity envelope needed to locate pinned schemas."""
    _schema_version_for_model(run, "M5_RUN")
    try:
        immutable.verify_model_seal(run)
    except immutable.ImmutableArtifactError as exc:
        _fail(f"M5_RUN bootstrap canonical model seal: {exc}")
    identities: dict[str, dict[str, Any]] = {}
    for index, raw_identity in enumerate(
        _exact_array(run.get("identities"), "RUN bootstrap identities")
    ):
        label = f"RUN bootstrap identities[{index}]"
        identity = _exact_object(raw_identity, label)
        _require_keys(identity, ("identity_id", "identity_kind", "version", "artifact"), label)
        identity_id = _safe_id(identity["identity_id"], f"{label}.identity_id")
        if identity_id in identities:
            _fail(f"duplicate bootstrap identity_id: {identity_id}")
        _exact_string(identity["identity_kind"], f"{label}.identity_kind")
        _exact_string(identity["version"], f"{label}.version")
        artifact = _exact_object(identity["artifact"], f"{label}.artifact")
        _require_keys(
            artifact,
            ("artifact_id", "relative_path", "sha256", "size_bytes", "media_type"),
            f"{label}.artifact",
        )
        _safe_id(artifact["artifact_id"], f"{label}.artifact.artifact_id")
        try:
            immutable.safe_relative_path(
                _exact_string(
                    artifact["relative_path"], f"{label}.artifact.relative_path"
                )
            )
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
        digest = _exact_string(artifact["sha256"], f"{label}.artifact.sha256")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            _fail(f"{label}.artifact.sha256 must be lowercase 64-hex")
        if _exact_int(artifact["size_bytes"], f"{label}.artifact.size_bytes") < 0:
            _fail(f"{label}.artifact.size_bytes must be nonnegative")
        if not _exact_string(artifact["media_type"], f"{label}.artifact.media_type"):
            _fail(f"{label}.artifact.media_type must not be empty")
        identities[identity_id] = identity
    return identities


def _pinned_schema_map(
    run: Mapping[str, Any], loader: _DeclaredSnapshotLoader
) -> dict[str, dict[str, Any]]:
    version = _schema_version_for_model(run, "M5_RUN")
    identities = _bootstrap_identity_map(run)
    binding_fields = {
        "M5_RUN": ("run_identity_id", "m5-run-schema"),
        "M5_RESULT": ("result_identity_id", "m5-result-schema"),
        "M5_DONE": ("done_identity_id", "m5-done-schema"),
    }
    bindings = _exact_object(run.get("schema_bindings"), "RUN schema_bindings")
    _require_keys(bindings, ("run_identity_id", "result_identity_id", "done_identity_id"), "RUN schema_bindings")
    schemas: dict[str, dict[str, Any]] = {}
    for kind, (field, expected_identity_id) in binding_fields.items():
        identity_id = _exact_string(bindings.get(field), f"schema_bindings.{field}")
        if identity_id != expected_identity_id:
            _fail(f"pinned {kind} schema binding is not the exact registered identity")
        identity = identities.get(identity_id)
        if identity is None or identity.get("identity_kind") != "SCHEMA":
            _fail(f"pinned {kind} schema identity is absent or wrong-kind")
        reference = identity["artifact"]
        if reference["sha256"] != SCHEMA_SHA256_BY_VERSION[version][kind]:
            _fail(f"pinned {kind} schema digest is not controller-pinned v{version}")
        payload = loader(reference)
        schema = _schema_from_bytes(payload, f"pinned {kind} schema")
        if schema.get("$id") != SCHEMA_IDS_BY_VERSION[version][kind]:
            _fail(f"pinned {kind} schema has the wrong explicit v{version} schema identifier")
        schemas[kind] = schema
    return schemas


def _verify_run_bundle(
    value: Path,
) -> tuple[
    Path,
    dict[str, Any],
    bytes,
    dict[str, dict[str, Any]],
    _DeclaredSnapshotLoader,
]:
    root, relative_path = _bundle_path(value, "RUN.json", "RUN bundle")
    loader = _DeclaredSnapshotLoader(root=root)
    try:
        run, raw = loader.read_manifest(relative_path, "M5 RUN")
        _require_pretty_bytes(run, raw, "M5 RUN")
        schemas = _pinned_schema_map(run, loader)
        _validate_run_semantics(run, schemas["M5_RUN"], loader)
        expected = [
            immutable.ImmutableFile(
                relative_path,
                immutable.sha256_bytes(raw),
                len(raw),
                source_path=root / relative_path,
            )
        ]
        expected.extend(
            immutable.ImmutableFile(
                reference["relative_path"],
                reference["sha256"],
                reference["size_bytes"],
                source_path=root / reference["relative_path"],
            )
            for reference in _run_artifacts(run).values()
        )
        try:
            immutable.verify_published_files(root, expected)
        except immutable.ImmutableArtifactError as exc:
            _fail(f"RUN bundle verification failed: {exc}")
        loader.verify_root()
        return root, run, raw, schemas, loader
    except BaseException:
        loader.close()
        raise


def _transform_evidence(
    value: Any,
    *,
    quantity_id: str,
    label: str,
    ledger: _AuthenticationLedger,
) -> dict[str, list[dict[str, Any]]]:
    evidence = _exact_object(value, label)
    _require_keys(evidence, EVIDENCE_ROLES, label)
    result: dict[str, list[dict[str, Any]]] = {}
    for role in EVIDENCE_ROLES:
        rows = _exact_array(evidence[role], f"{label}.{role}")
        if not rows:
            _fail(f"{label}.{role} must preserve at least one full artifact")
        result[role] = [
            ledger.add(
                row,
                role=f"EVIDENCE_{role.upper()}",
                context_id=f"{quantity_id}.{role}.{index}",
                label=f"{label}.{role}[{index}]",
            )
            for index, row in enumerate(rows)
        ]
    return result


def _normalize_outcome(value: Any, label: str) -> dict[str, Any]:
    outcome = copy.deepcopy(_exact_object(value, label))
    if outcome.get("kind") == "MIXTURE_UNRESOLVED":
        outcome["kind"] = "MIXTURE"
    return outcome


def _exact_finite_number(value: Any, label: str) -> float:
    if type(value) not in {int, float}:
        _fail(f"{label} must be an exact JSON number")
    if type(value) is int and abs(value) > _MAX_SAFE_JSON_INTEGER:
        _fail(f"{label} integer exceeds the exact safe range")
    converted = float(value)
    if not math.isfinite(converted):
        _fail(f"{label} must be finite")
    return converted


def _quantity_number(quantity: Mapping[str, Any], value: Any, label: str) -> float:
    number = _exact_finite_number(value, label)
    domain = quantity["value_domain"]
    kind = domain["kind"]
    if kind == "POSITIVE" and number <= 0:
        _fail(f"{label} must be positive")
    if kind == "NONNEGATIVE" and number < 0:
        _fail(f"{label} must be nonnegative")
    if kind == "PROBABILITY" and not 0.0 <= number <= 1.0:
        _fail(f"{label} must be in [0,1]")
    if domain["integral"] and not number.is_integer():
        _fail(f"{label} must be integral for its simulator consumer")
    return number


def _allowed_lineage_bindings(
    run: Mapping[str, Any], quantity: Mapping[str, Any]
) -> tuple[set[str], set[str], dict[str, str]]:
    if quantity["producer"]["kind"] == "PROBE_FAMILY":
        owner = next(
            family
            for family in run["families"]
            if family["family_id"] == quantity["producer"]["family_id"]
        )
    else:
        owner = run["components"]["pchase"]
    relationship_ids = set(owner["build_relationship_ids"])
    binary_by_relationship = {
        relationship["relationship_id"]: relationship["binary_identity_id"]
        for relationship in run["build_relationships"]
        if relationship["relationship_id"] in relationship_ids
    }
    if set(binary_by_relationship) != relationship_ids:
        _fail(f"quantity {quantity['quantity_id']} has incomplete build lineage")
    return set(owner["harness_identity_ids"]), relationship_ids, binary_by_relationship


def _quantity_dependency_contract(quantity: Mapping[str, Any]) -> str:
    upstream_family_ids = quantity["upstream_family_ids"]
    if not upstream_family_ids:
        return "NONE"
    return "UPSTREAM_FAMILIES:" + ",".join(upstream_family_ids)


def _expected_upstream_decision_sha256s(
    decision_hashes_by_family: Mapping[str, Iterable[str]],
    quantity: Mapping[str, Any],
) -> list[str]:
    """Derive the authority-declared scientific dependency for one quantity."""

    missing = [
        family_id
        for family_id in quantity["upstream_family_ids"]
        if not list(decision_hashes_by_family.get(family_id, ()))
    ]
    if missing:
        _fail(
            f"quantity {quantity['quantity_id']} lacks prior decisions for authority-declared families {missing}"
        )
    return sorted(
        decision_sha256
        for family_id in quantity["upstream_family_ids"]
        for decision_sha256 in decision_hashes_by_family[family_id]
    )


def _expected_source_lineage_id(
    run: Mapping[str, Any],
    quantity: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    harness_identity_id: str,
    build_relationship_id: str,
    binary_identity_id: str,
    capture_session_id: str,
) -> str:
    digest = immutable.sha256_bytes(
        immutable.canonical_json_bytes(
            {
                "run_model_seal_sha256": run["model_seal_sha256"],
                "quantity_id": quantity["quantity_id"],
                "readout_id": contract["readout_id"],
                "method_class": contract["method_class"],
                "instrument": contract["instrument"],
                "harness_identity_id": harness_identity_id,
                "build_relationship_id": build_relationship_id,
                "binary_identity_id": binary_identity_id,
                "capture_session_id": capture_session_id,
            }
        )
    )
    return f"lineage.{digest}"


def _derive_readouts(
    quantity: Mapping[str, Any],
    value: Any,
    evidence: Mapping[str, list[Mapping[str, Any]]],
    *,
    upstream_decision_sha256s: list[str],
    ledger: _AuthenticationLedger,
    loader: _DeclaredSnapshotLoader,
) -> tuple[list[dict[str, Any]], list[str]]:
    inputs = _exact_array(value, f"{quantity['quantity_id']} readout_inputs")
    contracts = quantity["readout_contracts"]
    if len(inputs) != len(contracts):
        _fail(f"{quantity['quantity_id']} must supply the exact readout producer roster")
    evidence_ids = {
        artifact["artifact_id"]
        for role in EVIDENCE_ROLES
        for artifact in evidence[role]
    }
    readouts: list[dict[str, Any]] = []
    repetition_ids: list[str] = []
    source_lineages: set[str] = set()
    capture_sessions: set[str] = set()
    harness_ids, relationship_ids, binary_by_relationship = _allowed_lineage_bindings(
        ledger._run, quantity
    )
    for index, (contract, authenticated_pair) in enumerate(zip(contracts, inputs, strict=True)):
        context = contract["readout_id"]
        pair = _exact_object(authenticated_pair, f"{context} authenticated inputs")
        _require_keys(pair, ("raw_sample", "producer"), f"{context} authenticated inputs")
        raw_reference = ledger.add(
            pair["raw_sample"],
            role="RAW_SAMPLE_SERIES",
            context_id=context,
            label=f"{quantity['quantity_id']} raw sample[{index}]",
        )
        reference = ledger.add(
            pair["producer"],
            role="READOUT_PRODUCER",
            context_id=context,
            label=f"{quantity['quantity_id']} readout producer[{index}]",
        )
        raw_sample_bytes = loader(raw_reference)
        try:
            raw_sample = immutable.strict_json_loads(
                raw_sample_bytes, label=f"raw sample series {context}"
            )
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
        raw_sample = _exact_object(raw_sample, f"raw sample series {context}")
        _require_keys(
            raw_sample,
            (
                "schema_version",
                "artifact_kind",
                "run_model_seal_sha256",
                "quantity_id",
                "readout_id",
                "method_class",
                "instrument",
                "source_lineage_id",
                "capture_session_id",
                "capture_started_unix_us",
                "device_uuid_sha256",
                "harness_identity_id",
                "build_relationship_id",
                "binary_identity_id",
                "upstream_decision_sha256s",
                "dependency_contract",
                "observations",
                "ncu_ancestor_artifact_ids",
                "simulator_ancestor_artifact_ids",
            ),
            f"raw sample series {context}",
        )
        lineage_id = _safe_id(
            raw_sample["source_lineage_id"], f"raw sample {context} source lineage"
        )
        capture_session_id = _safe_id(
            raw_sample["capture_session_id"], f"raw sample {context} capture session"
        )
        if capture_session_id in capture_sessions:
            _fail(f"quantity {quantity['quantity_id']} reuses a capture session")
        capture_sessions.add(capture_session_id)
        capture_started_unix_us = _exact_int(
            raw_sample["capture_started_unix_us"], f"raw sample {context} capture start"
        )
        harness_identity_id = raw_sample["harness_identity_id"]
        relationship_id = raw_sample["build_relationship_id"]
        binary_identity_id = raw_sample["binary_identity_id"]
        expected_lineage_id = _expected_source_lineage_id(
            ledger._run,
            quantity,
            contract,
            harness_identity_id=harness_identity_id,
            build_relationship_id=relationship_id,
            binary_identity_id=binary_identity_id,
            capture_session_id=capture_session_id,
        )
        if lineage_id in source_lineages:
            _fail(f"quantity {quantity['quantity_id']} reuses a raw-source lineage")
        source_lineages.add(lineage_id)
        expected_raw = {
            "schema_version": 1,
            "artifact_kind": "M5_RAW_SAMPLE_SERIES",
            "run_model_seal_sha256": ledger._run["model_seal_sha256"],
            "quantity_id": quantity["quantity_id"],
            "readout_id": contract["readout_id"],
            "method_class": contract["method_class"],
            "instrument": contract["instrument"],
        }
        if (
            any(raw_sample[key] != expected for key, expected in expected_raw.items())
            or lineage_id != expected_lineage_id
            or capture_started_unix_us < 0
            or raw_sample["device_uuid_sha256"]
            != ledger._run["target"]["device_uuid_sha256"]
            or raw_sample["upstream_decision_sha256s"] != upstream_decision_sha256s
            or raw_sample["dependency_contract"] != _quantity_dependency_contract(quantity)
            or harness_identity_id not in harness_ids
            or relationship_id not in relationship_ids
            or binary_identity_id != binary_by_relationship.get(relationship_id)
            or raw_sample["ncu_ancestor_artifact_ids"] != []
            or raw_sample["simulator_ancestor_artifact_ids"] != []
        ):
            _fail(f"raw sample series {context} has stale, NCU, simulator, or foreign lineage")
        raw = loader(reference)
        try:
            producer = immutable.strict_json_loads(raw, label=f"readout producer {context}")
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
        producer = _exact_object(producer, f"readout producer {context}")
        _require_keys(
            producer,
            (
                "schema_version",
                "artifact_kind",
                "run_model_seal_sha256",
                "quantity_id",
                "readout_id",
                "method_class",
                "instrument",
                "derivation_identity_id",
                "derivation_operation",
                "unit",
                "input_class",
                "sample_operation",
                "raw_sample_artifact_sha256",
                "source_lineage_id",
                "evidence_artifact_ids",
                "ncu_artifact_ids",
                "simulator_artifact_ids",
            ),
            f"readout producer {context}",
        )
        expected_fields = {
            "schema_version": 1,
            "artifact_kind": "M5_READOUT_PRODUCER",
            "run_model_seal_sha256": ledger._run["model_seal_sha256"],
            "quantity_id": quantity["quantity_id"],
            "readout_id": contract["readout_id"],
            "method_class": contract["method_class"],
            "instrument": contract["instrument"],
            "derivation_identity_id": contract["derivation_identity_id"],
            "derivation_operation": contract["derivation_operation"],
            "unit": quantity["unit"],
            "input_class": contract["instrument"],
            "sample_operation": contract["sample_operation"],
            "raw_sample_artifact_sha256": raw_reference["sha256"],
            "source_lineage_id": lineage_id,
        }
        if any(producer[key] != expected for key, expected in expected_fields.items()):
            _fail(f"readout producer {context} changed its preregistered contract")
        if producer["ncu_artifact_ids"] != [] or producer["simulator_artifact_ids"] != []:
            _fail(f"readout producer {context} violates Rule-1 input separation")
        observations = _exact_array(
            raw_sample["observations"], f"raw sample series {context} observations"
        )
        if not 3 <= len(observations) <= 4096:
            _fail(f"readout producer {context} observation count is outside 3..4096")
        values: list[float] = []
        producer_repetitions: list[str] = []
        launch_ids: list[str] = []
        observed_unix_us: list[int] = []
        for observation_index, raw_observation in enumerate(observations):
            observation_label = f"{context} observation[{observation_index}]"
            observation = _exact_object(raw_observation, observation_label)
            _require_keys(
                observation,
                ("repetition_id", "launch_id", "observed_unix_us", "numerator", "denominator"),
                observation_label,
            )
            repetition_id = _safe_id(
                observation["repetition_id"], f"{observation_label}.repetition_id"
            )
            launch_ids.append(
                _safe_id(observation["launch_id"], f"{observation_label}.launch_id")
            )
            observed = _exact_int(
                observation["observed_unix_us"], f"{observation_label}.observed_unix_us"
            )
            if observed <= capture_started_unix_us:
                _fail(f"{observation_label} does not follow its capture-session start")
            observed_unix_us.append(observed)
            numerator = _exact_finite_number(
                observation["numerator"], f"{observation_label}.numerator"
            )
            denominator = _exact_finite_number(
                observation["denominator"], f"{observation_label}.denominator"
            )
            if numerator < 0 or denominator <= 0 or max(abs(numerator), denominator) > 1.0e15:
                _fail(f"{observation_label} ratio inputs are outside the closed domain")
            sample = numerator / denominator
            if not math.isfinite(sample):
                _fail(f"{observation_label} ratio derivation overflowed")
            values.append(_quantity_number(quantity, sample, f"{observation_label} derived sample"))
            producer_repetitions.append(repetition_id)
        if (
            len(producer_repetitions) != len(set(producer_repetitions))
            or len(launch_ids) != len(set(launch_ids))
            or observed_unix_us != sorted(set(observed_unix_us))
        ):
            _fail(f"readout producer {context} repeats a protocol repetition or launch")
        if producer["sample_operation"] != "RATIO" or producer["derivation_operation"] not in {"MEDIAN", "MEAN"}:
            _fail(f"unsupported readout derivation contract for {context}")
        if producer["derivation_operation"] == "MEDIAN":
            ordered = sorted(values)
            middle = len(ordered) // 2
            derived = (
                ordered[middle]
                if len(ordered) % 2
                else ordered[middle - 1] / 2.0 + ordered[middle] / 2.0
            )
        else:
            derived = sum(values) / len(values)
        if not math.isfinite(derived):
            _fail(f"readout producer {context} derivation overflowed")
        derived = _quantity_number(quantity, derived, f"readout producer {context} median")
        bound_evidence = _exact_array(
            producer["evidence_artifact_ids"], f"{context} evidence IDs"
        )
        if len(bound_evidence) != len(set(bound_evidence)) or not set(bound_evidence).issubset(
            evidence_ids
        ):
            _fail(f"readout producer {context} has invalid retained-evidence bindings")
        for role in contract["required_evidence_roles"]:
            role_ids = {artifact["artifact_id"] for artifact in evidence[role]}
            if not set(bound_evidence).intersection(role_ids):
                _fail(f"readout producer {context} lacks required {role} evidence")
        if not producer_repetitions:
            _fail(f"readout producer {context} repetition IDs are empty")
        for repetition_id in producer_repetitions:
            repetition_ids.append(repetition_id)
        readouts.append(
            {
                "readout_id": contract["readout_id"],
                "method_class": contract["method_class"],
                "instrument": contract["instrument"],
                "readout_role": contract["readout_role"],
                "derivation_identity_id": contract["derivation_identity_id"],
                "derivation_operation": contract["derivation_operation"],
                "value": derived,
                "unit": quantity["unit"],
                "sample_count": len(values),
                "sample_values": values,
                "repetition_ids": list(producer_repetitions),
                "launch_ids": launch_ids,
                "observed_unix_us": observed_unix_us,
                "source_lineage_id": lineage_id,
                "capture_session_id": capture_session_id,
                "capture_started_unix_us": capture_started_unix_us,
                "upstream_decision_sha256s": list(upstream_decision_sha256s),
                "dependency_contract": _quantity_dependency_contract(quantity),
                "evidence_artifact_ids": list(bound_evidence),
                "raw_sample_artifact": raw_reference,
                "producer_artifact": reference,
                "ncu_used": False,
                "simulator_used": False,
            }
        )
    if len(repetition_ids) != len(set(repetition_ids)):
        _fail(f"quantity {quantity['quantity_id']} reuses repetitions across independent readouts")
    return readouts, repetition_ids


def _gap_diagnostics(
    quantity: Mapping[str, Any],
    readouts: list[Mapping[str, Any]],
    evidence: Mapping[str, list[Mapping[str, Any]]],
    kind: str,
) -> dict[str, Any]:
    """Return a retained, machine-checkable explanation for every typed gap."""
    values = [float(readout["value"]) for readout in readouts]
    observed = max(
        (
            _relative_delta(values[left], values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ),
        default=0.0,
    )
    tolerance = float(quantity["agreement_limit_fraction"])
    disagreement = observed > tolerance
    return {
        "disagreement_statistics": {
            "basis": "MAX_RELATIVE_DELTA",
            "observed_max_relative_delta": observed,
            "tolerance_fraction": tolerance,
            "readout_ids": [readout["readout_id"] for readout in readouts],
        },
        "gap_trigger": "READOUT_DISAGREEMENT" if disagreement else f"PREREGISTERED_{kind}",
        "contradiction_evidence": (
            [
                evidence["confound"][0]["artifact_id"],
                evidence["composition"][0]["artifact_id"],
            ]
            if disagreement
            else []
        ),
    }


def _outcome_from_kind(
    quantity: Mapping[str, Any],
    readouts: list[Mapping[str, Any]],
    evidence: Mapping[str, list[Mapping[str, Any]]],
    kind: str,
) -> dict[str, Any]:
    values = [float(readout["value"]) for readout in readouts]
    if kind == "MEASURED":
        return _measured_outcome(quantity, readouts)
    if kind not in quantity["authorized_gap_kinds"]:
        _fail(f"quantity {quantity['quantity_id']} did not preregister gap {kind}")
    if kind == "RANGE":
        return {
            "kind": "RANGE",
            "lower": min(values),
            "upper": max(values),
            "basis": "CONTROLLER_MIN_MAX_OF_INDEPENDENT_READOUT_MEDIANS",
            **_gap_diagnostics(quantity, readouts, evidence, kind),
        }
    if kind == "CENSORED":
        return {
            "kind": "CENSORED",
            "direction": "LOWER",
            "bound": min(values),
            "mechanism": "CONTROLLER_DERIVED_LOWER_BOUND_FROM_PREREGISTERED_READOUTS",
            "binder": quantity["consumer_binding"],
            **_gap_diagnostics(quantity, readouts, evidence, kind),
        }
    if kind == "MIXTURE":
        weight = 1.0 / len(readouts)
        return {
            "kind": "MIXTURE",
            "components": [
                {
                    "label": readout["readout_id"],
                    "value_lower": readout["value"],
                    "value_upper": readout["value"],
                    "weight_lower": weight,
                    "weight_upper": weight,
                }
                for readout in readouts
            ],
            "identifiability": "PREREGISTERED_READOUT_COMPONENTS_NOT_IDENTIFIED_BEYOND_EQUAL_WEIGHT",
            **_gap_diagnostics(quantity, readouts, evidence, kind),
        }
    if kind == "UNRESOLVED_STRUCTURE":
        confound = evidence["confound"][0]["artifact_id"]
        composition = evidence["composition"][0]["artifact_id"]
        return {
            "kind": "UNRESOLVED_STRUCTURE",
            "resolved_domain": quantity["conditioning"]["scope"],
            "candidates": [
                {
                    "candidate_id": _canonical_safe_id("candidate", readout["readout_id"]),
                    "parameterization": f"{readout['readout_id']}={readout['value']:.17g}",
                    "bound": quantity["consumer_binding"],
                }
                for readout in readouts
            ],
            "contradictions": [
                {
                    "contradiction_id": _canonical_safe_id(
                        "contradiction", quantity["quantity_id"]
                    ),
                    "description": "INDEPENDENT_PREREGISTERED_READOUTS_DO_NOT_SELECT_ONE_STRUCTURE",
                    "confound_details": "BOUND_TO_RETAINED_CONFOUND_ARTIFACT",
                    "composition_details": "BOUND_TO_RETAINED_COMPOSITION_ARTIFACT",
                    "artifact_ids": [confound, composition],
                }
            ],
            "consumer_conservative_rule": "RETAIN_ALL_CANDIDATES_WITHOUT_NCU_SELECTION",
            **_gap_diagnostics(quantity, readouts, evidence, kind),
        }
    if kind == "SURFACE_GAP":
        return {
            "kind": "SURFACE_GAP",
            "dimensions": [
                {"name": name, "domain": str(quantity["conditioning"][name])}
                for name in ("residency", "access_class", "scope", "variant")
            ],
            "binder_verdict": "PREREGISTERED_SURFACE_NOT_CLOSED_BY_RAW_SAMPLE_LINEAGES",
            "model_mismatch": quantity["consumer_binding"],
            **_gap_diagnostics(quantity, readouts, evidence, kind),
        }
    _fail(f"unsupported controller outcome kind: {kind}")


def _derive_outcome(
    quantity: Mapping[str, Any],
    readouts: list[Mapping[str, Any]],
    evidence: Mapping[str, list[Mapping[str, Any]]],
    authenticated: Any,
    *,
    ledger: _AuthenticationLedger,
    loader: _DeclaredSnapshotLoader,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = ledger.add(
        authenticated,
        role="QUANTITY_DECISION",
        context_id=quantity["quantity_id"],
        label=f"{quantity['quantity_id']} decision",
    )
    raw = loader(reference)
    try:
        producer = immutable.strict_json_loads(raw, label=f"decision {quantity['quantity_id']}")
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    producer = _exact_object(producer, f"decision {quantity['quantity_id']}")
    _require_keys(
        producer,
        (
            "schema_version",
            "artifact_kind",
            "run_model_seal_sha256",
            "quantity_id",
            "readout_producer_sha256s",
            "outcome",
            "ncu_artifact_ids",
            "simulator_artifact_ids",
        ),
        f"decision {quantity['quantity_id']}",
    )
    expected_hashes = [readout["producer_artifact"]["sha256"] for readout in readouts]
    if (
        producer["schema_version"] != 1
        or producer["artifact_kind"] != "M5_QUANTITY_DECISION"
        or producer["run_model_seal_sha256"] != ledger._run["model_seal_sha256"]
        or producer["quantity_id"] != quantity["quantity_id"]
        or producer["readout_producer_sha256s"] != expected_hashes
        or producer["ncu_artifact_ids"] != []
        or producer["simulator_artifact_ids"] != []
    ):
        _fail(f"decision {quantity['quantity_id']} is stale or violates Rule-1")
    declaration = _normalize_outcome(
        producer["outcome"], f"decision {quantity['quantity_id']} outcome"
    )
    if set(declaration) != {"kind"} or declaration["kind"] not in OUTCOME_KINDS:
        _fail("quantity decision may select only one preregistered outcome kind")
    outcome = _outcome_from_kind(
        quantity, readouts, evidence, declaration["kind"]
    )
    return outcome, reference


def _transform_cross_checks(
    value: Any,
    *,
    quantity_id: str,
    label: str,
    ledger: _AuthenticationLedger,
    loader: _DeclaredSnapshotLoader,
    candidate_model_seal_sha256: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(_exact_array(value, label)):
        row_label = f"{label}[{index}]"
        row = _exact_object(raw, row_label)
        _require_keys(row, ("cross_check_id", "ncu_artifact", "verdict_artifact"), row_label)
        cross_check_id = _safe_id(row["cross_check_id"], f"{row_label}.cross_check_id")
        ncu_reference = ledger.add(
            row["ncu_artifact"],
            role="NCU_RAW",
            context_id=cross_check_id,
            label=f"{row_label}.ncu_artifact",
        )
        verdict_reference = ledger.add(
            row["verdict_artifact"],
            role="NCU_VERDICT",
            context_id=cross_check_id,
            label=f"{row_label}.verdict_artifact",
        )
        verdict_raw = loader(verdict_reference)
        try:
            verdict = immutable.strict_json_loads(verdict_raw, label=f"{row_label} verdict")
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
        verdict = _exact_object(verdict, f"{row_label} verdict")
        _require_keys(
            verdict,
            (
                "schema_version",
                "artifact_kind",
                "run_model_seal_sha256",
                "quantity_id",
                "cross_check_id",
                "ncu_artifact_sha256",
                "candidate_model_seal_sha256",
                "verdict",
            ),
            f"{row_label} verdict",
        )
        if (
            verdict["schema_version"] != 1
            or verdict["artifact_kind"] != "M5_NCU_VERDICT"
            or verdict["run_model_seal_sha256"] != ledger._run["model_seal_sha256"]
            or verdict["quantity_id"] != quantity_id
            or verdict["cross_check_id"] != cross_check_id
            or verdict["ncu_artifact_sha256"] != ncu_reference["sha256"]
            or verdict["candidate_model_seal_sha256"] != candidate_model_seal_sha256
            or verdict["verdict"]
            not in {"CONSISTENT", "INCONSISTENT", "INCONCLUSIVE", "NOT_APPLICABLE"}
        ):
            _fail(f"{row_label} NCU verdict artifact is stale or malformed")
        result.append(
            {
                "cross_check_id": cross_check_id,
                "instrument": "NCU",
                "artifact": ncu_reference,
                "verdict_artifact": verdict_reference,
                "verdict": verdict["verdict"],
                "fitted": False,
                "derived": False,
                "selected": False,
                "tuned": False,
                "transcribed": False,
            }
        )
    return result


def _decode_control_word(high_word_hex: str, label: str) -> dict[str, int]:
    if type(high_word_hex) is not str or re.fullmatch(r"0x[0-9a-f]{16}", high_word_hex) is None:
        _fail(f"{label} high_word must be lowercase 0x plus 16 hex digits")
    control_bits = (int(high_word_hex, 16) >> 41) & 0x7FFFFF
    if (control_bits >> 21) & 0x3:
        _fail(f"{label} decoded record has nonzero reserved control bits")
    return {
        "stall": control_bits & 0xF,
        "yield_flag": (control_bits >> 4) & 0x1,
        "write_barrier": (control_bits >> 5) & 0x7,
        "read_barrier": (control_bits >> 8) & 0x7,
        "wait_mask": (control_bits >> 11) & 0x3F,
        "reuse": (control_bits >> 17) & 0xF,
    }


def _parse_sidecar(payload: bytes, label: str) -> dict[str, Any]:
    """Validate and rederive the real emit_control_bits_sidecar.py contract."""
    try:
        value = immutable.strict_json_loads(payload, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    sidecar = _exact_object(value, label)
    _require_keys(
        sidecar,
        (
            "arch",
            "functions",
            "skipped_arch_sections",
            "parser_contract",
            "toolkit_version",
            "binary",
            "summary",
        ),
        label,
    )
    if sidecar["arch"] != "sm_86" or sidecar["parser_contract"] != CONTROL_PARSER_CONTRACT:
        _fail(f"{label} changed the pinned extractor arch/parser contract")
    toolkit_version = _exact_string(sidecar["toolkit_version"], f"{label}.toolkit_version")
    if re.fullmatch(r"V13\.2(?:\.[0-9]+)?", toolkit_version) is None:
        _fail(f"{label} was not produced by the pinned V13.2 toolkit")
    if not _exact_string(sidecar["binary"], f"{label}.binary"):
        _fail(f"{label}.binary must preserve the producer-observed path")
    skipped = _exact_array(sidecar["skipped_arch_sections"], f"{label}.skipped_arch_sections")
    if skipped != sorted(set(skipped)) or any(
        type(arch) is not str or arch == "sm_86" for arch in skipped
    ):
        _fail(f"{label} skipped-architecture inventory is malformed")
    functions = _exact_object(sidecar["functions"], f"{label}.functions")
    if not functions:
        _fail(f"{label} contains no SASS functions")
    function_summary: dict[str, dict[str, int]] = {}
    total_decoded = 0
    total_unknown = 0
    for function_name, raw_records in functions.items():
        if type(function_name) is not str or not function_name or len(function_name) > 4096:
            _fail(f"{label} contains an invalid function name")
        records = _exact_array(raw_records, f"{label}.functions[{function_name!r}]")
        if len(records) > 1_000_000:
            _fail(f"{label} function record count exceeds the closed bound")
        seen_pcs: set[int] = set()
        decoded = 0
        unknown = 0
        static_stalls = 0
        for index, raw_record in enumerate(records):
            record_label = f"{label}.{function_name}[{index}]"
            record = _exact_object(raw_record, record_label)
            status = record.get("status")
            common = {"pc", "predicate", "opcode_text", "low_word", "status"}
            if status == "decoded":
                _require_keys(
                    record,
                    (*common, "high_word", "control"),
                    record_label,
                )
            elif status == "unknown_encoding":
                permitted = common | {"high_word", "reason"}
                if set(record) not in (common | {"reason"}, permitted):
                    _fail(f"{record_label} unknown record has an unregistered shape")
                if not _exact_string(record.get("reason"), f"{record_label}.reason"):
                    _fail(f"{record_label} unknown record lacks a typed reason")
            else:
                _fail(f"{record_label} has no closed extractor status")
            pc = _exact_int(record.get("pc"), f"{record_label}.pc")
            if pc < 0 or pc % 16 or pc in seen_pcs:
                _fail(f"{record_label} PC is negative, unaligned, or duplicated")
            seen_pcs.add(pc)
            predicate = record.get("predicate")
            if predicate is not None and type(predicate) is not str:
                _fail(f"{record_label}.predicate must be string or null")
            if not _exact_string(record.get("opcode_text"), f"{record_label}.opcode_text"):
                _fail(f"{record_label} has empty opcode text")
            if type(record.get("low_word")) is not str or re.fullmatch(
                r"0x[0-9a-f]{16}", record["low_word"]
            ) is None:
                _fail(f"{record_label} low_word is not canonical")
            if status == "decoded":
                expected_control = _decode_control_word(record["high_word"], record_label)
                if record["control"] != expected_control:
                    _fail(f"{record_label} control fields were not derived from high_word")
                decoded += 1
                static_stalls += expected_control["stall"]
            else:
                if "high_word" in record and (
                    type(record["high_word"]) is not str
                    or re.fullmatch(r"0x[0-9a-f]{16}", record["high_word"]) is None
                ):
                    _fail(f"{record_label} unknown high_word is not canonical")
                unknown += 1
        total_decoded += decoded
        total_unknown += unknown
        function_summary[function_name] = {
            "instructions": len(records),
            "decoded": decoded,
            "unknown_encoding": unknown,
            "stall_cycles_static_sum": static_stalls,
        }
    expected_summary = {
        "functions": function_summary,
        "total_decoded": total_decoded,
        "total_unknown_encoding": total_unknown,
    }
    if sidecar["summary"] != expected_summary:
        _fail(f"{label} summary is not rederived from standard instruction records")
    return {
        "model": sidecar,
        "decoded_instruction_count": total_decoded,
        "unknown_encoding_count": total_unknown,
        "function_summary": function_summary,
    }


def _parse_pc_join_witness(
    payload: bytes,
    label: str,
    *,
    run: Mapping[str, Any],
    expected_members: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        value = immutable.strict_json_loads(payload, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    witness = _exact_object(value, label)
    _require_keys(
        witness,
        ("schema_version", "artifact_kind", "arch", "source_commit", "n_cells", "n_full_join", "cells"),
        label,
    )
    if (
        witness["schema_version"] != 1
        or witness["artifact_kind"] != "sass_control_pc_join_witness"
        or witness["arch"] != "sm_86"
        or witness["source_commit"] != run["source_state"]["source_commit"]
    ):
        _fail(f"{label} is stale or has the wrong producer contract")
    cells = _exact_array(witness["cells"], f"{label}.cells")
    if _exact_int(witness["n_cells"], f"{label}.n_cells") != len(cells):
        _fail(f"{label} n_cells is stale")
    expected_by_key = {
        (member["slug"], member["shape_point"], member["binary_sha256"]): member
        for member in expected_members
    }
    if len(expected_by_key) != len(expected_members):
        _fail(f"{label} expected witness keys are not unique")
    result: dict[str, dict[str, Any]] = {}
    full_count = 0
    closed_cell_reasons = {"binary_or_trace_unresolved"}
    closed_function_reasons = {
        "kernel_sass_not_in_cell_binary",
        "kernel_function_match_ambiguous",
    }
    for index, raw_cell in enumerate(cells):
        cell_label = f"{label}.cells[{index}]"
        cell = _exact_object(raw_cell, cell_label)
        identity = _exact_object(cell.get("identity"), f"{cell_label}.identity")
        required_identity = {"slug", "shape_point", "binary_sha256"}
        optional_identity = {
            "variant_id",
            "kernel_signature_sha256",
            "arch",
            "source_commit",
        }
        if not required_identity.issubset(identity) or not set(identity).issubset(
            required_identity | optional_identity
        ):
            _fail(f"{cell_label} identity is not the standard witness identity")
        if identity.get("arch", "sm_86") != "sm_86" or identity.get(
            "source_commit", run["source_state"]["source_commit"]
        ) != run["source_state"]["source_commit"]:
            _fail(f"{cell_label} identity arch/source commit is stale")
        key = (identity["slug"], identity["shape_point"], identity["binary_sha256"])
        member = expected_by_key.get(key)
        if member is None or member["cell_id"] in result:
            _fail(f"{cell_label} is unexpected or duplicated")
        if "kernel_signature_sha256" in identity and re.fullmatch(
            r"[0-9a-f]{64}", str(identity["kernel_signature_sha256"])
        ) is None:
            _fail(f"{cell_label} kernel signature is malformed")
        if cell.get("status") == "typed":
            _require_keys(cell, ("identity", "status", "typed_reason"), cell_label)
            if cell["typed_reason"] not in closed_cell_reasons:
                _fail(f"{cell_label} has an unregistered typed reason")
            normalized = {
                "full_join": False,
                "typed_reasons": [cell["typed_reason"]],
                "function_rows": [],
                "coverage": {"n_traced_pcs": 0, "n_joined": 0, "n_decoded": 0, "full_join": False},
            }
        else:
            _require_keys(
                cell,
                ("identity", "first_launch", "vpc_low32_mismatches", "n_instruction_records", "functions", "coverage"),
                cell_label,
            )
            if not _exact_string(cell["first_launch"], f"{cell_label}.first_launch"):
                _fail(f"{cell_label} lacks a first launch")
            mismatches = _exact_int(cell["vpc_low32_mismatches"], f"{cell_label}.vpc_low32_mismatches")
            instruction_records = _exact_int(cell["n_instruction_records"], f"{cell_label}.n_instruction_records")
            if mismatches < 0 or instruction_records < 0:
                _fail(f"{cell_label} has negative witness counts")
            function_rows = _exact_array(cell["functions"], f"{cell_label}.functions")
            traced = joined = decoded = 0
            typed_reasons: list[str] = []
            function_ids: set[int] = set()
            for function_index, raw_function in enumerate(function_rows):
                function_label = f"{cell_label}.functions[{function_index}]"
                function = _exact_object(raw_function, function_label)
                function_id = _exact_int(function.get("function_unique_id"), f"{function_label}.function_unique_id")
                if function_id in function_ids:
                    _fail(f"{cell_label} duplicates a function_unique_id")
                function_ids.add(function_id)
                if not _exact_string(function.get("kernel_name"), f"{function_label}.kernel_name"):
                    _fail(f"{function_label} has no kernel name")
                if function.get("status") == "typed":
                    _require_keys(
                        function,
                        ("function_unique_id", "kernel_name", "status", "typed_reason", "detail", "n_matches"),
                        function_label,
                    )
                    if function["typed_reason"] not in closed_function_reasons:
                        _fail(f"{function_label} has an unregistered typed reason")
                    if _exact_int(function["n_matches"], f"{function_label}.n_matches") < 0:
                        _fail(f"{function_label} has a negative match count")
                    typed_reasons.append(function["typed_reason"])
                elif function.get("status") == "joined":
                    _require_keys(
                        function,
                        ("function_unique_id", "kernel_name", "sass_function", "status", "n_traced_pcs", "n_joined", "n_decoded", "missing_pcs", "sample"),
                        function_label,
                    )
                    if not _exact_string(function["sass_function"], f"{function_label}.sass_function"):
                        _fail(f"{function_label} has no joined SASS function")
                    counts = [
                        _exact_int(function[name], f"{function_label}.{name}")
                        for name in ("n_traced_pcs", "n_joined", "n_decoded")
                    ]
                    if any(count < 0 for count in counts) or not counts[0] >= counts[1] >= counts[2]:
                        _fail(f"{function_label} has impossible join counts")
                    missing = _exact_array(function["missing_pcs"], f"{function_label}.missing_pcs")
                    if len(missing) > 8 or any(type(pc) is not int or pc < 0 for pc in missing):
                        _fail(f"{function_label} missing-PC sample is malformed")
                    if function["sample"] is not None and type(function["sample"]) is not dict:
                        _fail(f"{function_label} sample must be object or null")
                    traced += counts[0]
                    joined += counts[1]
                    decoded += counts[2]
                else:
                    _fail(f"{function_label} has no standard join status")
            coverage = _exact_object(cell["coverage"], f"{cell_label}.coverage")
            _require_keys(coverage, ("n_traced_pcs", "n_joined", "n_decoded", "full_join"), f"{cell_label}.coverage")
            expected_full = traced == joined == decoded and traced > 0 and not typed_reasons and mismatches == 0
            expected_coverage = {
                "n_traced_pcs": traced,
                "n_joined": joined,
                "n_decoded": decoded,
                "full_join": expected_full,
            }
            if coverage != expected_coverage:
                _fail(f"{cell_label} coverage is not rederived from function rows")
            if expected_full:
                full_count += 1
            normalized = {
                "full_join": expected_full,
                "typed_reasons": sorted(set(typed_reasons)),
                "function_rows": function_rows,
                "coverage": coverage,
            }
        result[member["cell_id"]] = normalized
    if set(result) != {member["cell_id"] for member in expected_members}:
        _fail(f"{label} does not close the exact planned member roster")
    if _exact_int(witness["n_full_join"], f"{label}.n_full_join") != full_count:
        _fail(f"{label} n_full_join is stale")
    return result


def _derived_control_entry(
    member: Mapping[str, Any],
    coverage_class: str,
    sidecar_reference: Mapping[str, Any],
    witness_reference: Mapping[str, Any],
    sidecar_facts: Mapping[str, Any],
    witness_cell: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = list(witness_cell["typed_reasons"])
    if sidecar_facts["decoded_instruction_count"] == 0:
        reasons.append("no_decoded_instructions")
    if sidecar_facts["unknown_encoding_count"]:
        reasons.append("unknown_encoding")
    if not witness_cell["full_join"] and not reasons:
        reasons.append("pc_join_incomplete")
    sidecar_functions = sidecar_facts["function_summary"]
    sidecar_records = sidecar_facts["model"]["functions"]
    for function in witness_cell["function_rows"]:
        if function.get("status") != "joined":
            continue
        summary = sidecar_functions.get(function["sass_function"])
        if summary is None or function["n_decoded"] > summary["decoded"]:
            _fail(f"control witness for {member['cell_id']} does not join its retained sidecar")
        sample = function["sample"]
        if sample is not None:
            sample = _exact_object(sample, f"control sample for {member['cell_id']}")
            _require_keys(
                sample,
                ("pc", "opcode_text", "control"),
                f"control sample for {member['cell_id']}",
            )
            matching = [
                record
                for record in sidecar_records[function["sass_function"]]
                if record["pc"] == sample["pc"]
            ]
            if len(matching) != 1:
                _fail(f"control witness sample PC is absent for {member['cell_id']}")
            retained = matching[0]
            if (
                retained["status"] != "decoded"
                or sample["opcode_text"] != retained["opcode_text"][:60]
                or sample["control"] != retained["control"]
            ):
                _fail(f"control witness sample differs from sidecar bytes for {member['cell_id']}")
    entry = {
        "coverage_class": coverage_class,
        "cell_id": member["cell_id"],
        "binary_sha256": member["binary_sha256"],
        "expected_sidecar_id": member["expected_sidecar_id"],
        "status": "COMPLETE" if not reasons else "TYPED",
        "decoded_instruction_count": sidecar_facts["decoded_instruction_count"],
        "unknown_encoding_count": sidecar_facts["unknown_encoding_count"],
        "pc_join_passed": witness_cell["full_join"],
        "function_join_passed": witness_cell["full_join"],
        "parser_contract": copy.deepcopy(CONTROL_PARSER_CONTRACT),
        "sidecar": dict(sidecar_reference),
        "pc_join_witness": dict(witness_reference),
    }
    if reasons:
        entry["typed_reasons"] = sorted(set(reasons))
    return entry


def _transform_control_group(
    value: Any,
    *,
    group_id: str,
    expected_members: list[Mapping[str, Any]],
    expected_witness_id: str,
    run: Mapping[str, Any],
    ledger: _AuthenticationLedger,
    loader: _DeclaredSnapshotLoader,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group = _exact_object(value, f"analysis control group {group_id}")
    _require_keys(group, ("pc_join_witness", "entries"), f"analysis control group {group_id}")
    witness_reference = ledger.add(
        group["pc_join_witness"],
        role="CONTROL_PC_JOIN_WITNESS",
        context_id=group_id,
        label=f"analysis {group_id} PC-join witness",
    )
    if witness_reference["artifact_id"] != expected_witness_id:
        _fail(f"analysis {group_id} PC-join witness artifact ID differs from RUN")
    witness_cells = _parse_pc_join_witness(
        loader(witness_reference),
        f"analysis {group_id} PC-join witness",
        run=run,
        expected_members=expected_members,
    )
    raw_entries = _exact_array(group["entries"], f"analysis {group_id} sidecars")
    if len(raw_entries) != len(expected_members):
        _fail(f"analysis {group_id} sidecar count differs from RUN")
    entries: list[dict[str, Any]] = []
    for index, (raw, member) in enumerate(zip(raw_entries, expected_members, strict=True)):
        label = f"analysis {group_id} sidecars[{index}]"
        row = _exact_object(raw, label)
        _require_keys(row, ("cell_id", "expected_sidecar_id", "sidecar"), label)
        if (
            row["cell_id"] != member["cell_id"]
            or row["expected_sidecar_id"] != member["expected_sidecar_id"]
        ):
            _fail(f"{label} changed its RUN-bound member identity")
        reference = ledger.add(
            row["sidecar"],
            role="CONTROL_SIDECAR",
            context_id=member["cell_id"],
            label=f"{label}.sidecar",
        )
        if reference["artifact_id"] != member["expected_sidecar_id"]:
            _fail(f"{label} sidecar artifact ID mismatch")
        facts = _parse_sidecar(loader(reference), f"standard sidecar {member['cell_id']}")
        entries.append(
            _derived_control_entry(
                member,
                group_id,
                reference,
                witness_reference,
                facts,
                witness_cells[member["cell_id"]],
            )
        )
    return entries, witness_reference


def _transform_controls(
    value: Any,
    *,
    run: Mapping[str, Any],
    material: dict[str, immutable.ImmutableFile],
    loader: _DeclaredSnapshotLoader,
    ledger: _AuthenticationLedger,
) -> dict[str, Any]:
    component = _exact_object(value, "analysis roster_control_sidecars")
    _require_keys(component, ("canonical", "calibration"), "analysis roster_control_sidecars")
    planned = run["components"]["roster_control_sidecars"]
    canonical_members = run["components"]["canonical_runnable_roster"]["members"]
    calibration_members = planned["calibration_members"]
    canonical_entries, canonical_witness = _transform_control_group(
        component["canonical"],
        group_id="CANONICAL_ROSTER",
        expected_members=canonical_members,
        expected_witness_id=planned["expected_canonical_pc_join_witness_id"],
        run=run,
        ledger=ledger,
        loader=loader,
    )
    calibration_entries, calibration_witness = _transform_control_group(
        component["calibration"],
        group_id="CALIBRATION_BINARIES",
        expected_members=calibration_members,
        expected_witness_id=planned["expected_calibration_pc_join_witness_id"],
        run=run,
        ledger=ledger,
        loader=loader,
    )
    all_entries = canonical_entries + calibration_entries
    inventory_unsigned = {
        "schema_version": 1,
        "artifact_kind": "M5_CONTROL_SIDECAR_INVENTORY",
        "run_model_seal_sha256": run["model_seal_sha256"],
        "parser_contract": copy.deepcopy(CONTROL_PARSER_CONTRACT),
        "canonical_pc_join_witness": canonical_witness,
        "calibration_pc_join_witness": calibration_witness,
        "entries": [
            {
                "coverage_class": entry["coverage_class"],
                "cell_id": entry["cell_id"],
                "status": entry["status"],
                "decoded_instruction_count": entry["decoded_instruction_count"],
                "unknown_encoding_count": entry["unknown_encoding_count"],
                "binary_sha256": entry["binary_sha256"],
                "pc_join_passed": entry["pc_join_passed"],
                "function_join_passed": entry["function_join_passed"],
                "sidecar_artifact_id": entry["sidecar"]["artifact_id"],
                "sidecar_sha256": entry["sidecar"]["sha256"],
                **(
                    {"typed_reasons": entry["typed_reasons"]}
                    if entry["status"] == "TYPED"
                    else {}
                ),
            }
            for entry in all_entries
        ],
    }
    inventory_model = immutable.seal_model(inventory_unsigned)
    inventory_bytes = immutable.pretty_json_bytes(inventory_model)
    digest = immutable.sha256_bytes(inventory_bytes)
    reference = _artifact_reference(
        planned["expected_inventory_id"], digest, len(inventory_bytes), "application/json"
    )
    _merge_material(
        material, immutable.ImmutableFile.from_bytes(reference["relative_path"], inventory_bytes)
    )
    return {
        "component_id": "roster_control_sidecars",
        "parser_contract": copy.deepcopy(CONTROL_PARSER_CONTRACT),
        "inventory_artifact": reference,
        "canonical_pc_join_witness": canonical_witness,
        "calibration_pc_join_witness": calibration_witness,
        "expected_canonical_cell_count": len(canonical_members),
        "observed_canonical_cell_count": len(canonical_entries),
        "expected_calibration_binary_count": len(calibration_members),
        "observed_calibration_binary_count": len(calibration_entries),
        "entries": all_entries,
    }


def _capture_observation_index(
    quantity_results: list[Mapping[str, Any]],
    execution_order: Iterable[str] = EXECUTION_ORDER,
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    capture_sessions: set[str] = set()
    launch_ids: set[str] = set()
    family_windows: dict[str, list[int]] = defaultdict(list)
    for row in quantity_results:
        producer = row["producer"]
        family = (
            producer["family_id"] if producer["kind"] == "PROBE_FAMILY" else "pchase"
        )
        for readout in row["readouts"]:
            session = readout["capture_session_id"]
            if session in capture_sessions:
                _fail("independent readouts reuse one capture session")
            capture_sessions.add(session)
            for repetition_id, launch_id, observed_at in zip(
                readout["repetition_ids"],
                readout["launch_ids"],
                readout["observed_unix_us"],
                strict=True,
            ):
                if repetition_id in observations or launch_id in launch_ids:
                    _fail("capture chronology reuses a repetition or launch globally")
                launch_ids.add(launch_id)
                observations[repetition_id] = {
                    "launch_id": launch_id,
                    "observed_unix_us": observed_at,
                    "family_id": family,
                    "capture_session_id": session,
                }
                if family != "pchase":
                    family_windows[family].append(observed_at)
    if set(family_windows) != set(execution_order):
        _fail("capture chronology does not cover all authority probe families")
    previous_max: int | None = None
    for family in execution_order:
        window = family_windows[family]
        current_min = min(window)
        current_max = max(window)
        if previous_max is not None and current_min <= previous_max:
            _fail("capture chronology violates b,a,d,f,c,e execution order")
        previous_max = current_max
    return observations


def _validate_protocol_state_witness(
    payload: bytes,
    *,
    run: Mapping[str, Any],
    capture_observations: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    label = "M5 protocol-state witness"
    try:
        value = immutable.strict_json_loads(payload, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    witness = _exact_object(value, label)
    _require_keys(
        witness,
        (
            "schema_version",
            "artifact_kind",
            "run_model_seal_sha256",
            "source_commit",
            "protocol_identity_id",
            "m0",
            "global_pre",
            "global_post",
            "repetitions",
        ),
        label,
    )
    if (
        witness["schema_version"] != 1
        or witness["artifact_kind"] != "M5_PROTOCOL_STATE_WITNESS"
        or witness["run_model_seal_sha256"] != run["model_seal_sha256"]
        or witness["source_commit"] != run["source_state"]["source_commit"]
        or witness["protocol_identity_id"] != "m5-campaign-protocol"
    ):
        _fail("M5 protocol-state witness is stale or cross-RUN")

    seen_observations: set[str] = set()

    def state(raw: Any, *, phase: str, repetition_id: str | None) -> dict[str, Any]:
        row = _exact_object(raw, f"{label} {phase}")
        _require_keys(
            row,
            (
                "observation_id",
                "observation_phase",
                "repetition_id",
                "observed_unix_us",
                "device_uuid_sha256",
                "sm_clock_mhz",
                "memory_clock_mhz",
                "persistence_mode",
                "clock_locked",
                "cache_control",
                "driver_version",
                "nvcc_release",
                "ncu_version",
                "ncu_replay_mode",
                "throttle_reasons",
            ),
            f"{label} {phase}",
        )
        observation_id = _safe_id(row["observation_id"], "protocol observation_id")
        if observation_id in seen_observations:
            _fail("protocol witness reuses an observation ID")
        seen_observations.add(observation_id)
        observed_unix_us = _exact_int(
            row["observed_unix_us"], f"protocol witness {phase} observation time"
        )
        if (
            observed_unix_us < 0
            or row["observation_phase"] != phase
            or row["repetition_id"] != repetition_id
            or row["device_uuid_sha256"] != run["target"]["device_uuid_sha256"]
            or type(row["sm_clock_mhz"]) is not int
            or row["sm_clock_mhz"] != 1350
            or type(row["memory_clock_mhz"]) is not int
            or row["memory_clock_mhz"] != 10501
            or row["persistence_mode"] is not True
            or row["clock_locked"] is not True
            or row["cache_control"] != "all"
            or row["driver_version"] != "610.43.02"
            or row["nvcc_release"] != "13.3"
            or row["ncu_version"] != "2026.1.0"
            or row["ncu_replay_mode"] != "kernel"
            or row["throttle_reasons"] != []
        ):
            _fail(f"protocol witness {phase} state differs from the complete v2.1 lock")
        return row

    m0_state = state(witness["m0"], phase="M0", repetition_id=None)
    global_pre_state = state(witness["global_pre"], phase="PRE", repetition_id=None)
    global_post_state = state(witness["global_post"], phase="POST", repetition_id=None)
    expected_repetitions = set(capture_observations)
    if not (
        m0_state["observed_unix_us"] < global_pre_state["observed_unix_us"]
        < global_post_state["observed_unix_us"]
    ):
        _fail("protocol M0/global PRE/global POST chronology is invalid")
    observed_repetitions: set[str] = set()
    rows = _exact_array(witness["repetitions"], f"{label}.repetitions")
    for index, raw in enumerate(rows):
        repetition = _exact_object(raw, f"{label}.repetitions[{index}]")
        _require_keys(repetition, ("repetition_id", "pre", "post"), f"{label}.repetitions[{index}]")
        repetition_id = _safe_id(repetition["repetition_id"], "protocol repetition_id")
        if repetition_id in observed_repetitions:
            _fail("protocol witness duplicates a repetition state join")
        observed_repetitions.add(repetition_id)
        pre_state = state(repetition["pre"], phase="PRE", repetition_id=repetition_id)
        post_state = state(repetition["post"], phase="POST", repetition_id=repetition_id)
        capture = capture_observations.get(repetition_id)
        if capture is None or not (
            global_pre_state["observed_unix_us"]
            < pre_state["observed_unix_us"]
            < capture["observed_unix_us"]
            < post_state["observed_unix_us"]
            < global_post_state["observed_unix_us"]
        ):
            _fail("protocol repetition PRE/capture/POST chronology is invalid")
    if observed_repetitions != expected_repetitions:
        _fail("protocol witness does not exactly join all readout producer repetitions")
    return {
        "component_id": "protocol_state",
        "protocol_identity_id": "m5-campaign-protocol",
        "witness_artifact": dict(reference),
        "m0_observation_count": 1,
        "global_pre_observation_count": 1,
        "global_post_observation_count": 1,
        "repetition_count": len(observed_repetitions),
        "per_repetition_pre_count": len(observed_repetitions),
        "per_repetition_post_count": len(observed_repetitions),
        "all_state_joins_passed": True,
        "clock_pair": {"sm_clock_mhz": 1350, "memory_clock_mhz": 10501},
        "persistence_mode": True,
        "cache_control": "all",
        "driver_version": "610.43.02",
        "nvcc_release": "13.3",
        "ncu_version": "2026.1.0",
        "ncu_replay_mode": "kernel",
        "device_uuid_sha256": run["target"]["device_uuid_sha256"],
        "chronology_closed": True,
        "throttle_reasons": [],
    }


def _fresh_pchase_card(
    run: Mapping[str, Any],
    quantity_results: list[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    material: dict[str, immutable.ImmutableFile],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pchase = run["components"]["pchase"]
    by_id = {row["quantity_id"]: row for row in quantity_results}
    properties = []
    for quantity_id in pchase["quantity_ids"]:
        row = by_id.get(quantity_id)
        if row is None:
            _fail(f"fresh pchase card is missing {quantity_id}")
        properties.append(
            {
                "property_id": quantity_id.removeprefix("pchase."),
                "quantity_id": quantity_id,
                "unit": row["unit"],
                "outcome": copy.deepcopy(row["outcome"]),
                "readout_producer_sha256s": [
                    readout["producer_artifact"]["sha256"] for readout in row["readouts"]
                ],
                "evidence_artifact_ids": sorted(
                    artifact["artifact_id"]
                    for role in EVIDENCE_ROLES
                    for artifact in row["evidence"][role]
                ),
            }
        )
    if len(properties) != 14 or len({row["property_id"] for row in properties}) != 14:
        _fail("fresh pchase card does not preserve the exact 14-property roster")
    unsigned = {
        "schema_version": 1,
        "artifact_kind": "M5_PCHASE_CALIBRATION_CARD",
        "card_id": f"m5-pchase-sm86-{run['model_seal_sha256']}",
        "run_model_seal_sha256": run["model_seal_sha256"],
        "source_commit": run["source_state"]["source_commit"],
        "architecture": "sm_86",
        "sku": "RTX 3090 Ti",
        "protocol_witness_sha256": protocol["witness_artifact"]["sha256"],
        "runner_identity_id": "m5-controller",
        "source_identity_id": "m5-pchase-source",
        "historical_card_identity_id": "m5-pchase-v1-card",
        "build_relationship_ids": list(pchase["build_relationship_ids"]),
        "property_count": 14,
        "properties": properties,
        "architecture_card_transcription": False,
        "publication_mode": "ADDITIVE_EVIDENCE_ONLY",
    }
    card = immutable.seal_model(unsigned)
    payload = immutable.pretty_json_bytes(card)
    digest = immutable.sha256_bytes(payload)
    reference = _artifact_reference("m5-pchase-fresh-card", digest, len(payload), "application/json")
    _merge_material(material, immutable.ImmutableFile.from_bytes(reference["relative_path"], payload))
    return card, reference


def _pre_ncu_candidate_model(
    run: Mapping[str, Any],
    quantity_results: list[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    controls: Mapping[str, Any],
    fresh_pchase_card: Mapping[str, Any],
    feature_authenticity: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "artifact_kind": "M5_PRE_NCU_CANDIDATE",
        "run_model_seal_sha256": run["model_seal_sha256"],
        "source_commit": run["source_state"]["source_commit"],
        "ncu_attached": False,
        "frozen_before_ncu_input_reads": True,
        "quantity_candidates": [
            {
                "quantity_id": row["quantity_id"],
                "readouts": copy.deepcopy(row["readouts"]),
                "outcome": copy.deepcopy(row["outcome"]),
                "decision_artifact": copy.deepcopy(row["decision_artifact"]),
                "evidence": copy.deepcopy(row["evidence"]),
            }
            for row in quantity_results
        ],
        "protocol_state": copy.deepcopy(protocol),
        "control_inventory_artifact": copy.deepcopy(controls["inventory_artifact"]),
        "fresh_pchase_card": copy.deepcopy(fresh_pchase_card),
        "feature_authenticity": copy.deepcopy(feature_authenticity),
    }
    return immutable.seal_model(unsigned)


def _freeze_pre_ncu_candidate(
    run: Mapping[str, Any],
    quantity_results: list[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    controls: Mapping[str, Any],
    fresh_pchase_card: Mapping[str, Any],
    feature_authenticity: Mapping[str, Any],
    material: dict[str, immutable.ImmutableFile],
    *,
    publication_root: Path | None = None,
) -> dict[str, Any]:
    candidate = _pre_ncu_candidate_model(
        run,
        quantity_results,
        protocol,
        controls,
        fresh_pchase_card,
        feature_authenticity,
    )
    payload = immutable.pretty_json_bytes(candidate)
    digest = immutable.sha256_bytes(payload)
    reference = _artifact_reference("m5-pre-ncu-candidate", digest, len(payload), "application/json")
    _merge_material(material, immutable.ImmutableFile.from_bytes(reference["relative_path"], payload))
    publication = {
        "publication_identity_id": f"m5-pre-ncu-publication.{candidate['model_seal_sha256']}",
        "artifact": reference,
        "publication_name": f"m5-pre-ncu-{candidate['model_seal_sha256']}",
        "verified": False,
        "verification_method": "REHASHED_PUBLISHED_BYTES",
        "published_sha256": None,
        "published_size_bytes": None,
    }
    if publication_root is not None:
        root = _external_output_root(publication_root)
        final_name = f"m5-pre-ncu-{candidate['model_seal_sha256']}"
        candidate_file = material[reference["relative_path"]]

        def commit_guard(commit_root: Path, descriptor: int) -> None:
            _authorize_output_descriptor(commit_root, descriptor)
            _reauthorize_run_source_and_controller(run)

        try:
            published = immutable.publish_directory(
                root,
                final_name,
                (candidate_file,),
                before_commit=commit_guard,
            )
            published_bytes = immutable.read_regular(published / reference["relative_path"])
        except immutable.ImmutableArtifactError as exc:
            _fail(f"pre-NCU candidate publication failed: {exc}")
        if published_bytes != payload:
            _fail("pre-NCU candidate publication did not preserve exact candidate bytes")
        publication.update(
            {
                "verified": True,
                "published_sha256": immutable.sha256_bytes(published_bytes),
                "published_size_bytes": len(published_bytes),
            }
        )
    return {
        "component_id": "pre_ncu_candidate",
        "artifact": reference,
        "candidate_model_seal_sha256": candidate["model_seal_sha256"],
        "feature_authenticated_artifact_count": len(feature_authenticity["entries"]),
        "ncu_attachment_phase": "AFTER_IMMUTABLE_CANDIDATE_FREEZE",
        "candidate_publication": publication,
    }


def _analysis_request_to_result(
    request: Mapping[str, Any],
    request_path: Path,
    run: Mapping[str, Any],
    run_artifact_loader: Callable[[Mapping[str, Any]], bytes],
) -> tuple[
    dict[str, Any],
    dict[str, immutable.ImmutableFile],
    _DeclaredSnapshotLoader,
]:
    _require_keys(
        request,
        (
            "schema_version",
            "request_kind",
            "run_model_seal_sha256",
            "protocol_state_witness",
            "quantity_results",
            "roster_control_sidecars",
        ),
        "analysis request",
    )
    if _exact_int(request["schema_version"], "analysis schema_version") != ACTIVE_SCHEMA_VERSION:
        _fail("unsupported explicit analysis request schema_version: v2 is required")
    if request["request_kind"] != "M5_ANALYSIS_REQUEST":
        _fail("analysis request_kind must be M5_ANALYSIS_REQUEST")
    if request["run_model_seal_sha256"] != run["model_seal_sha256"]:
        _fail("analysis request does not bind the exact RUN model seal")
    material: dict[str, immutable.ImmutableFile] = {}
    loader = _DeclaredSnapshotLoader(material=material)
    ledger = _AuthenticationLedger(
        run,
        base=request_path.parent,
        material=material,
        loader=loader,
        run_artifact_loader=run_artifact_loader,
    )
    quantity_results: list[dict[str, Any]] = []
    repetition_ids: list[str] = []
    pending_cross_checks: list[Any] = []
    decision_hashes_by_family: dict[str, list[str]] = defaultdict(list)
    raw_results = _exact_array(request["quantity_results"], "quantity_results")
    if len(raw_results) != len(run["quantities"]):
        _fail("analysis request must contain the exact RUN quantity roster")
    for index, (raw, quantity) in enumerate(zip(raw_results, run["quantities"], strict=True)):
        label = f"analysis quantity_results[{index}]"
        row = _exact_object(raw, label)
        _require_keys(
            row,
            ("quantity_id", "evidence", "readout_inputs", "decision", "cross_checks"),
            label,
        )
        if row["quantity_id"] != quantity["quantity_id"]:
            _fail(f"{label} changed the RUN quantity order or identity")
        evidence = _transform_evidence(
            row["evidence"],
            quantity_id=quantity["quantity_id"],
            label=f"{label}.evidence",
            ledger=ledger,
        )
        producer_family = (
            quantity["producer"]["family_id"]
            if quantity["producer"]["kind"] == "PROBE_FAMILY"
            else "pchase"
        )
        upstream_decision_sha256s = _expected_upstream_decision_sha256s(
            decision_hashes_by_family, quantity
        )
        readouts, repetitions = _derive_readouts(
            quantity,
            row["readout_inputs"],
            evidence,
            upstream_decision_sha256s=upstream_decision_sha256s,
            ledger=ledger,
            loader=loader,
        )
        outcome, decision_reference = _derive_outcome(
            quantity,
            readouts,
            evidence,
            row["decision"],
            ledger=ledger,
            loader=loader,
        )
        repetition_ids.extend(repetitions)
        decision_hashes_by_family[producer_family].append(decision_reference["sha256"])
        pending_cross_checks.append(row["cross_checks"])
        quantity_results.append(
            {
                "quantity_id": quantity["quantity_id"],
                "quantity_group": quantity["quantity_group"],
                "meaning": quantity["meaning"],
                "producer": copy.deepcopy(quantity["producer"]),
                "consumer_binding": quantity["consumer_binding"],
                "conditioning": copy.deepcopy(quantity["conditioning"]),
                "unit": quantity["unit"],
                "readouts": readouts,
                "outcome": outcome,
                "decision_artifact": decision_reference,
                "evidence": evidence,
                "cross_checks": [],
            }
        )
    capture_observations = _capture_observation_index(
        quantity_results, run["family_execution_order"]
    )
    protocol_reference = ledger.add(
        request["protocol_state_witness"],
        role="PROTOCOL_STATE_WITNESS",
        context_id="protocol-state",
        label="analysis protocol_state_witness",
    )
    if protocol_reference["artifact_id"] != "m5-protocol-state-witness":
        _fail("protocol-state witness artifact ID is not the preregistered ID")
    protocol = _validate_protocol_state_witness(
        loader(protocol_reference),
        run=run,
        capture_observations=capture_observations,
        reference=protocol_reference,
    )
    controls = _transform_controls(
        request["roster_control_sidecars"],
        run=run,
        material=material,
        loader=loader,
        ledger=ledger,
    )
    pchase_ids = run["components"]["pchase"]["quantity_ids"]
    pchase_evidence = sorted(
        {
            artifact["artifact_id"]
            for row in quantity_results
            if row["quantity_id"] in pchase_ids
            for role in EVIDENCE_ROLES
            for artifact in row["evidence"][role]
        }
    )
    _card, fresh_card_reference = _fresh_pchase_card(
        run, quantity_results, protocol, material
    )
    feature_authenticity = ledger.model()
    if any(
        entry["artifact_role"] in {"NCU_RAW", "NCU_VERDICT"}
        for entry in feature_authenticity["entries"]
    ):
        _fail("NCU input was read before pre-NCU candidate freeze")
    pre_ncu_candidate = _freeze_pre_ncu_candidate(
        run,
        quantity_results,
        protocol,
        controls,
        fresh_card_reference,
        feature_authenticity,
        material,
        publication_root=request_path.parent / "m5-pre-ncu-candidate-publications",
    )
    # Only after the immutable candidate bytes exist may verdict-only NCU
    # artifacts be opened and authenticated.
    for index, (row, pending) in enumerate(
        zip(quantity_results, pending_cross_checks, strict=True)
    ):
        row["cross_checks"] = _transform_cross_checks(
            pending,
            quantity_id=row["quantity_id"],
            label=f"analysis quantity_results[{index}].cross_checks",
            ledger=ledger,
            loader=loader,
            candidate_model_seal_sha256=pre_ncu_candidate["candidate_model_seal_sha256"],
        )
    ncu_entries = [
        entry for entry in ledger.entries if entry["artifact_role"] in {"NCU_RAW", "NCU_VERDICT"}
    ]
    feature_entries = [
        entry for entry in ledger.entries if entry["artifact_role"] not in {"NCU_RAW", "NCU_VERDICT"}
    ]
    if (
        {entry["artifact"]["artifact_id"] for entry in ncu_entries}
        & {entry["artifact"]["artifact_id"] for entry in feature_entries}
        or {entry["artifact"]["sha256"] for entry in ncu_entries}
        & {entry["artifact"]["sha256"] for entry in feature_entries}
    ):
        _fail("Rule-1 forbids NCU artifact IDs or bytes from entering feature evidence")
    measured = sum(row["outcome"].get("kind") == "MEASURED" for row in quantity_results)
    evidence_count = sum(
        len(row["evidence"][role]) for row in quantity_results for role in EVIDENCE_ROLES
    )
    authenticity = ledger.model()
    unsigned = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "artifact_kind": "M5_RESULT",
        "run_model_seal_sha256": run["model_seal_sha256"],
        "source_commit": run["source_state"]["source_commit"],
        "family_execution_order": list(run["family_execution_order"]),
        "family_scientific_rank": list(run["family_scientific_rank"]),
        "quantity_results": quantity_results,
        "authenticity": authenticity,
        "components": {
            "pre_ncu_candidate": pre_ncu_candidate,
            "protocol_state": protocol,
            "pchase": {
                "component_id": "pchase",
                "quantity_ids": list(pchase_ids),
                "evidence_artifact_ids": pchase_evidence,
                "fresh_card_artifact": fresh_card_reference,
                "property_count": 14,
                "historical_card_identity_id": "m5-pchase-v1-card",
                "architecture_card_transcription": False,
            },
            "roster_control_sidecars": controls,
        },
        "counts": {
            "required_quantity_count": len(run["quantities"]),
            "result_quantity_count": len(quantity_results),
            "measured_quantity_count": measured,
            "typed_gap_quantity_count": len(quantity_results) - measured,
            "evidence_artifact_reference_count": evidence_count,
            "raw_sample_series_count": sum(len(row["readouts"]) for row in quantity_results),
            "pre_ncu_authenticated_artifact_count": pre_ncu_candidate[
                "feature_authenticated_artifact_count"
            ],
            "authenticated_artifact_count": len(authenticity["entries"]),
            "authority_receipt_count": len(authenticity["entries"]),
            "protocol_repetition_count": protocol["repetition_count"],
            "canonical_control_sidecar_count": controls["observed_canonical_cell_count"],
            "calibration_control_sidecar_count": controls["observed_calibration_binary_count"],
        },
    }
    return immutable.seal_model(unsigned), material, loader


def _relative_delta(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        _fail("relative-delta inputs must be finite")
    denominator = max(abs(left), abs(right))
    if denominator == 0:
        return 0.0
    numerator = abs(left - right)
    result = numerator / denominator
    if not math.isfinite(numerator) or not math.isfinite(result):
        _fail("relative-delta arithmetic overflowed")
    return result


def _evidence_ids_by_role(row: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        role: {artifact["artifact_id"] for artifact in row["evidence"][role]}
        for role in EVIDENCE_ROLES
    }


def _validate_readouts(quantity: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    contracts = quantity["readout_contracts"]
    readouts = row["readouts"]
    if len(readouts) != len(contracts) or len(readouts) > 16:
        _fail(f"result {quantity['quantity_id']} does not carry the exact planned readout count")
    evidence_by_role = _evidence_ids_by_role(row)
    evidence_hashes = {
        artifact["artifact_id"]: artifact["sha256"]
        for role in EVIDENCE_ROLES
        for artifact in row["evidence"][role]
    }
    all_evidence = set().union(*evidence_by_role.values())
    prior_evidence_sets: list[set[str]] = []
    prior_evidence_hash_sets: list[set[str]] = []
    raw_lineages: set[str] = set()
    raw_hashes: set[str] = set()
    capture_sessions: set[str] = set()
    for index, (contract, readout) in enumerate(zip(contracts, readouts, strict=True)):
        for key in (
            "readout_id",
            "method_class",
            "instrument",
            "readout_role",
            "derivation_identity_id",
            "derivation_operation",
        ):
            if readout[key] != contract[key]:
                _fail(f"result {quantity['quantity_id']} readout[{index}] changed planned {key}")
        if readout["unit"] != quantity["unit"]:
            _fail(f"result {quantity['quantity_id']} readout unit mismatch")
        _quantity_number(quantity, readout["value"], f"result {quantity['quantity_id']} readout value")
        repetitions = readout["repetition_ids"]
        launches = readout["launch_ids"]
        observed_times = readout["observed_unix_us"]
        capture_session = _safe_id(readout["capture_session_id"], "result capture session")
        capture_started = _exact_int(
            readout["capture_started_unix_us"], "result capture start"
        )
        lineage = _safe_id(readout["source_lineage_id"], "result raw source lineage")
        raw_hash = readout["raw_sample_artifact"]["sha256"]
        if (
            readout["sample_count"] != len(repetitions)
            or readout["sample_count"] != len(readout["sample_values"])
            or readout["sample_count"] != len(launches)
            or readout["sample_count"] != len(observed_times)
            or len(repetitions) < 3
            or len(repetitions) != len(set(repetitions))
            or len(launches) != len(set(launches))
            or observed_times != sorted(set(observed_times))
            or any(time <= capture_started for time in observed_times)
            or capture_session in capture_sessions
            or lineage in raw_lineages
            or raw_hash in raw_hashes
            or readout["ncu_used"] is not False
            or readout["simulator_used"] is not False
            or readout["producer_artifact"]["size_bytes"] < 1
            or readout["raw_sample_artifact"]["size_bytes"] < 1
        ):
            _fail(f"result {quantity['quantity_id']} readout producer closure is malformed")
        raw_lineages.add(lineage)
        raw_hashes.add(raw_hash)
        capture_sessions.add(capture_session)
        for sample in readout["sample_values"]:
            _quantity_number(quantity, sample, f"result {quantity['quantity_id']} sample")
        bound = set(readout["evidence_artifact_ids"])
        if not bound or not bound.issubset(all_evidence):
            _fail(f"result {quantity['quantity_id']} readout is not bound to retained evidence")
        for role in contract["required_evidence_roles"]:
            if not bound.intersection(evidence_by_role[role]):
                _fail(f"result {quantity['quantity_id']} readout lacks required {role} evidence")
        bound_hashes = {evidence_hashes[artifact_id] for artifact_id in bound}
        if any(bound.intersection(previous) for previous in prior_evidence_sets) or any(
            bound_hashes.intersection(previous) for previous in prior_evidence_hash_sets
        ):
            _fail(
                f"result {quantity['quantity_id']} independent readouts reuse evidence bytes"
            )
        prior_evidence_sets.append(bound)
        prior_evidence_hash_sets.append(bound_hashes)


def _validate_outcome(quantity: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    outcome = row["outcome"]
    kind = outcome["kind"]
    if kind not in OUTCOME_KINDS:
        _fail(f"unapproved outcome token: {kind!r}")
    if kind in TYPED_GAPS and kind not in quantity["authorized_gap_kinds"]:
        _fail(f"quantity {quantity['quantity_id']} emitted unplanned gap kind {kind}")
    if kind in TYPED_GAPS:
        required_gap_fields = {
            "disagreement_statistics",
            "gap_trigger",
            "contradiction_evidence",
        }
        if not required_gap_fields.issubset(outcome):
            _fail(f"{kind} {quantity['quantity_id']} lacks typed gap diagnostics")
        values = [float(readout["value"]) for readout in row["readouts"]]
        observed = max(
            (
                _relative_delta(values[left], values[right])
                for left in range(len(values))
                for right in range(left + 1, len(values))
            ),
            default=0.0,
        )
        statistics = outcome["disagreement_statistics"]
        if not math.isclose(
            observed,
            float(statistics["observed_max_relative_delta"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or statistics["readout_ids"] != [
            readout["readout_id"] for readout in row["readouts"]
        ]:
            _fail(f"{kind} {quantity['quantity_id']} disagreement statistics are stale")
        if outcome["gap_trigger"] == "READOUT_DISAGREEMENT" and observed <= float(
            quantity["agreement_limit_fraction"]
        ):
            _fail(f"{kind} {quantity['quantity_id']} declares disagreement below tolerance")
        if outcome["gap_trigger"] == "READOUT_DISAGREEMENT":
            evidence_by_role = _evidence_ids_by_role(row)
            contradiction = set(outcome["contradiction_evidence"])
            if not contradiction.intersection(evidence_by_role["confound"]) or not contradiction.intersection(
                evidence_by_role["composition"]
            ):
                _fail(f"{kind} {quantity['quantity_id']} disagreement lacks contradiction evidence")
    if kind == "MEASURED":
        value_readouts = [
            readout for readout in row["readouts"] if readout["readout_role"] == "VALUE"
        ]
        if len(value_readouts) < 2:
            _fail(f"MEASURED {quantity['quantity_id']} requires two VALUE readouts")
        agreement = outcome["agreement"]
        expected_witness = _methodological_independence_witness(quantity, row["readouts"])
        if agreement.get("independence_witness") != expected_witness or agreement.get(
            "methodologically_independent"
        ) is not True:
            _fail(f"MEASURED {quantity['quantity_id']} lacks a derived independence witness")
        readout_ids = [readout["readout_id"] for readout in value_readouts]
        if agreement["readout_ids"] != readout_ids:
            _fail(f"MEASURED {quantity['quantity_id']} agreement must enumerate exact VALUE readouts")
        tolerance = agreement["tolerance_fraction"]
        if tolerance > quantity["agreement_limit_fraction"]:
            _fail(f"MEASURED {quantity['quantity_id']} exceeds planned agreement limit")
        values = [float(readout["value"]) for readout in value_readouts]
        observed = max(
            (
                _relative_delta(values[left], values[right])
                for left in range(len(values))
                for right in range(left + 1, len(values))
            ),
            default=0.0,
        )
        if not math.isclose(
            observed,
            float(agreement["observed_max_relative_delta"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            _fail(f"MEASURED {quantity['quantity_id']} agreement statistic is stale")
        if observed > tolerance:
            _fail(f"MEASURED {quantity['quantity_id']} VALUE readouts disagree")
        measured = _quantity_number(
            quantity, outcome["value"], f"MEASURED {quantity['quantity_id']} value"
        )
        if any(_relative_delta(measured, value) > tolerance for value in values):
            _fail(f"MEASURED {quantity['quantity_id']} value lacks readout support")
    elif kind == "RANGE":
        lower = _quantity_number(quantity, outcome["lower"], f"RANGE {quantity['quantity_id']} lower")
        upper = _quantity_number(quantity, outcome["upper"], f"RANGE {quantity['quantity_id']} upper")
        if lower > upper:
            _fail(f"RANGE {quantity['quantity_id']} has inverted bounds")
    elif kind == "CENSORED":
        _quantity_number(quantity, outcome["bound"], f"CENSORED {quantity['quantity_id']} bound")
    elif kind == "MIXTURE":
        lower = upper = 0.0
        labels: set[str] = set()
        for component in outcome["components"]:
            value_lower = _quantity_number(
                quantity,
                component["value_lower"],
                f"MIXTURE {quantity['quantity_id']} value_lower",
            )
            value_upper = _quantity_number(
                quantity,
                component["value_upper"],
                f"MIXTURE {quantity['quantity_id']} value_upper",
            )
            if (
                value_lower > value_upper
                or component["weight_lower"] > component["weight_upper"]
            ):
                _fail(f"MIXTURE {quantity['quantity_id']} has inverted bounds")
            if component["label"] in labels:
                _fail(f"MIXTURE {quantity['quantity_id']} has duplicate labels")
            labels.add(component["label"])
            lower += component["weight_lower"]
            upper += component["weight_upper"]
        if lower > 1.0 + 1e-12 or upper < 1.0 - 1e-12:
            _fail(f"MIXTURE {quantity['quantity_id']} weights cannot compose to one")
    elif kind == "UNRESOLVED_STRUCTURE":
        candidate_ids = [candidate["candidate_id"] for candidate in outcome["candidates"]]
        contradiction_ids_seen = [
            contradiction["contradiction_id"] for contradiction in outcome["contradictions"]
        ]
        if len(candidate_ids) != len(set(candidate_ids)) or len(
            contradiction_ids_seen
        ) != len(set(contradiction_ids_seen)):
            _fail(f"UNRESOLVED_STRUCTURE {quantity['quantity_id']} has duplicate structure IDs")
        evidence_by_role = _evidence_ids_by_role(row)
        evidence = set().union(*evidence_by_role.values())
        for contradiction in outcome["contradictions"]:
            contradiction_ids = set(contradiction["artifact_ids"])
            if not contradiction_ids.issubset(evidence):
                _fail(f"UNRESOLVED_STRUCTURE {quantity['quantity_id']} contradiction lacks evidence")
            if not contradiction_ids.intersection(
                evidence_by_role["confound"]
            ) or not contradiction_ids.intersection(evidence_by_role["composition"]):
                _fail(
                    f"UNRESOLVED_STRUCTURE {quantity['quantity_id']} contradiction must bind confound and composition evidence"
                )


def _rederive_readout(
    run: Mapping[str, Any],
    quantity: Mapping[str, Any],
    contract: Mapping[str, Any],
    readout: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, Any]], bytes],
    *,
    upstream_decision_sha256s: list[str] | None = None,
) -> None:
    if upstream_decision_sha256s is None:
        upstream_decision_sha256s = []
    label = f"retained readout producer {contract['readout_id']}"
    raw_sample_bytes = artifact_loader(readout["raw_sample_artifact"])
    try:
        raw_sample = immutable.strict_json_loads(
            raw_sample_bytes, label=f"{label} raw sample series"
        )
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    raw_sample = _exact_object(raw_sample, f"{label} raw sample series")
    _require_keys(
        raw_sample,
        (
            "schema_version",
            "artifact_kind",
            "run_model_seal_sha256",
            "quantity_id",
            "readout_id",
            "method_class",
            "instrument",
            "source_lineage_id",
            "capture_session_id",
            "capture_started_unix_us",
            "device_uuid_sha256",
            "harness_identity_id",
            "build_relationship_id",
            "binary_identity_id",
            "upstream_decision_sha256s",
            "dependency_contract",
            "observations",
            "ncu_ancestor_artifact_ids",
            "simulator_ancestor_artifact_ids",
        ),
        f"{label} raw sample series",
    )
    harness_ids, relationship_ids, binary_by_relationship = _allowed_lineage_bindings(
        run, quantity
    )
    relationship_id = raw_sample["build_relationship_id"]
    harness_identity_id = raw_sample["harness_identity_id"]
    binary_identity_id = raw_sample["binary_identity_id"]
    capture_session_id = _safe_id(raw_sample["capture_session_id"], f"{label} capture session")
    capture_started_unix_us = _exact_int(
        raw_sample["capture_started_unix_us"], f"{label} capture start"
    )
    lineage_id = _safe_id(raw_sample["source_lineage_id"], f"{label} source lineage")
    expected_lineage_id = _expected_source_lineage_id(
        run,
        quantity,
        contract,
        harness_identity_id=harness_identity_id,
        build_relationship_id=relationship_id,
        binary_identity_id=binary_identity_id,
        capture_session_id=capture_session_id,
    )
    expected_raw = {
        "schema_version": 1,
        "artifact_kind": "M5_RAW_SAMPLE_SERIES",
        "run_model_seal_sha256": run["model_seal_sha256"],
        "quantity_id": quantity["quantity_id"],
        "readout_id": contract["readout_id"],
        "method_class": contract["method_class"],
        "instrument": contract["instrument"],
    }
    if (
        any(raw_sample[key] != expected for key, expected in expected_raw.items())
        or lineage_id != expected_lineage_id
        or capture_started_unix_us < 0
        or raw_sample["device_uuid_sha256"] != run["target"]["device_uuid_sha256"]
        or raw_sample["upstream_decision_sha256s"] != upstream_decision_sha256s
        or raw_sample["dependency_contract"] != _quantity_dependency_contract(quantity)
        or harness_identity_id not in harness_ids
        or relationship_id not in relationship_ids
        or binary_identity_id != binary_by_relationship.get(relationship_id)
        or raw_sample["ncu_ancestor_artifact_ids"] != []
        or raw_sample["simulator_ancestor_artifact_ids"] != []
    ):
        _fail(f"{label} raw sample lineage is stale, foreign, NCU, or simulator-derived")
    raw = artifact_loader(readout["producer_artifact"])
    try:
        producer = immutable.strict_json_loads(raw, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    producer = _exact_object(producer, label)
    _require_keys(
        producer,
        (
            "schema_version",
            "artifact_kind",
            "run_model_seal_sha256",
            "quantity_id",
            "readout_id",
            "method_class",
            "instrument",
            "derivation_identity_id",
            "derivation_operation",
            "unit",
            "input_class",
            "sample_operation",
            "raw_sample_artifact_sha256",
            "source_lineage_id",
            "evidence_artifact_ids",
            "ncu_artifact_ids",
            "simulator_artifact_ids",
        ),
        label,
    )
    expected_header = {
        "schema_version": 1,
        "artifact_kind": "M5_READOUT_PRODUCER",
        "run_model_seal_sha256": run["model_seal_sha256"],
        "quantity_id": quantity["quantity_id"],
        "readout_id": contract["readout_id"],
        "method_class": contract["method_class"],
        "instrument": contract["instrument"],
        "derivation_identity_id": contract["derivation_identity_id"],
        "derivation_operation": contract["derivation_operation"],
        "unit": quantity["unit"],
        "input_class": contract["instrument"],
        "sample_operation": contract["sample_operation"],
        "raw_sample_artifact_sha256": readout["raw_sample_artifact"]["sha256"],
        "source_lineage_id": lineage_id,
    }
    if any(producer[key] != expected for key, expected in expected_header.items()) or (
        producer["ncu_artifact_ids"] != [] or producer["simulator_artifact_ids"] != []
    ):
        _fail(f"{label} changed its closed derivation contract")
    observations = _exact_array(raw_sample["observations"], f"{label}.observations")
    if not 3 <= len(observations) <= 4096:
        _fail(f"{label} observation count is outside 3..4096")
    values: list[float] = []
    repetitions: list[str] = []
    launch_ids: list[str] = []
    observed_unix_us: list[int] = []
    for index, raw_observation in enumerate(observations):
        observation = _exact_object(raw_observation, f"{label}.observations[{index}]")
        _require_keys(
            observation,
            ("repetition_id", "launch_id", "observed_unix_us", "numerator", "denominator"),
            f"{label}.observations[{index}]",
        )
        repetitions.append(_safe_id(observation["repetition_id"], f"{label} repetition"))
        launch_ids.append(_safe_id(observation["launch_id"], f"{label} launch"))
        observed = _exact_int(observation["observed_unix_us"], f"{label} observed time")
        if observed <= capture_started_unix_us:
            _fail(f"{label} observation does not follow capture start")
        observed_unix_us.append(observed)
        numerator = _exact_finite_number(observation["numerator"], f"{label} numerator")
        denominator = _exact_finite_number(observation["denominator"], f"{label} denominator")
        if numerator < 0 or denominator <= 0 or max(abs(numerator), denominator) > 1.0e15:
            _fail(f"{label} ratio input is outside the closed domain")
        values.append(_quantity_number(quantity, numerator / denominator, f"{label} sample"))
    if (
        len(repetitions) != len(set(repetitions))
        or len(launch_ids) != len(set(launch_ids))
        or observed_unix_us != sorted(set(observed_unix_us))
    ):
        _fail(f"{label} duplicates a repetition, launch, or observation time")
    if producer["sample_operation"] != "RATIO" or producer["derivation_operation"] not in {"MEDIAN", "MEAN"}:
        _fail(f"unsupported readout derivation contract for {label}")
    if producer["derivation_operation"] == "MEDIAN":
        ordered = sorted(values)
        middle = len(ordered) // 2
        derived = (
            ordered[middle]
            if len(ordered) % 2
            else ordered[middle - 1] / 2.0 + ordered[middle] / 2.0
        )
    else:
        derived = sum(values) / len(values)
    derived = _quantity_number(quantity, derived, f"{label} median")
    expected_readout = {
        "readout_id": contract["readout_id"],
        "method_class": contract["method_class"],
        "instrument": contract["instrument"],
        "readout_role": contract["readout_role"],
        "derivation_identity_id": contract["derivation_identity_id"],
        "derivation_operation": contract["derivation_operation"],
        "value": derived,
        "unit": quantity["unit"],
        "sample_count": len(values),
        "sample_values": values,
        "repetition_ids": repetitions,
        "launch_ids": launch_ids,
        "observed_unix_us": observed_unix_us,
        "source_lineage_id": lineage_id,
        "capture_session_id": capture_session_id,
        "capture_started_unix_us": capture_started_unix_us,
        "upstream_decision_sha256s": list(upstream_decision_sha256s),
        "dependency_contract": _quantity_dependency_contract(quantity),
        "evidence_artifact_ids": producer["evidence_artifact_ids"],
        "raw_sample_artifact": readout["raw_sample_artifact"],
        "producer_artifact": readout["producer_artifact"],
        "ncu_used": False,
        "simulator_used": False,
    }
    if readout != expected_readout:
        _fail(f"RESULT readout {contract['readout_id']} is not rederived from retained producer bytes")


def _methodological_independence_witness(
    quantity: Mapping[str, Any], readouts: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Derive a falsifiable independence witness from the sealed readouts."""
    if len(readouts) < 2:
        _fail(f"{quantity['quantity_id']} needs two readouts for independence")
    contracts = {
        contract["readout_id"]: contract for contract in quantity["readout_contracts"]
    }
    selected = [contracts.get(readout["readout_id"]) for readout in readouts]
    if any(contract is None for contract in selected):
        _fail(f"{quantity['quantity_id']} readout has no preregistered method contract")
    methods = [contract["method_contract"] for contract in selected if contract is not None]
    estimator_forms = {
        (contract["sample_operation"], contract["derivation_operation"], method["estimator_identity"])
        for contract, method in zip(selected, methods, strict=True)
    }
    mechanisms = {method["mechanism"] for method in methods}
    confounder_sets = [set(method["confounder_set"]) for method in methods]
    raw_hashes = [readout["raw_sample_artifact"]["sha256"] for readout in readouts]
    lineages = [readout["source_lineage_id"] for readout in readouts]
    sessions = [readout["capture_session_id"] for readout in readouts]
    repetitions = [set(readout["repetition_ids"]) for readout in readouts]
    launches = [set(readout["launch_ids"]) for readout in readouts]
    if len(estimator_forms) != len(readouts):
        _fail(f"{quantity['quantity_id']} readouts reuse the same estimator form")
    if len(mechanisms) != len(readouts):
        _fail(f"{quantity['quantity_id']} readouts reuse the same mechanism")
    if len(set(raw_hashes)) != len(raw_hashes) or len(set(lineages)) != len(lineages) or len(set(sessions)) != len(sessions):
        _fail(f"{quantity['quantity_id']} readouts overlap raw ancestry")
    if any(repetitions[left].intersection(repetitions[right]) or launches[left].intersection(launches[right]) for left in range(len(readouts)) for right in range(left + 1, len(readouts))):
        _fail(f"{quantity['quantity_id']} readouts overlap repetition or launch ancestry")
    if any(left.intersection(right) for left in confounder_sets for right in confounder_sets if left is not right):
        _fail(f"{quantity['quantity_id']} readouts share confounder controls")
    falsification_rules = {method["falsification_rule"] for method in methods}
    if len(falsification_rules) != 1:
        _fail(f"{quantity['quantity_id']} readouts do not share a preregistered falsification rule")
    return {
        "witness_version": 1,
        "readout_ids": [readout["readout_id"] for readout in readouts],
        "raw_ancestry_disjoint": True,
        "estimator_forms_distinct": True,
        "mechanisms_distinct": True,
        "confounder_sets_disjoint": True,
        "falsification_rule": next(iter(falsification_rules)),
    }


def _measured_outcome(
    quantity: Mapping[str, Any], readouts: list[Mapping[str, Any]]
) -> dict[str, Any]:
    values = [float(readout["value"]) for readout in readouts if readout["readout_role"] == "VALUE"]
    if len(values) < 2:
        _fail(f"MEASURED {quantity['quantity_id']} lacks two VALUE readouts")
    observed = max(
        (
            _relative_delta(values[left], values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ),
        default=0.0,
    )
    tolerance = float(quantity["agreement_limit_fraction"])
    independence_witness = _methodological_independence_witness(quantity, readouts)
    if observed > tolerance:
        _fail(f"MEASURED {quantity['quantity_id']} retained readouts disagree")
    measured = _quantity_number(
        quantity,
        sum(value / len(values) for value in values),
        f"MEASURED {quantity['quantity_id']} value",
    )
    return {
        "kind": "MEASURED",
        "value": measured,
        "agreement": {
            "explicit": True,
            "methodologically_independent": bool(
                independence_witness["raw_ancestry_disjoint"]
                and independence_witness["estimator_forms_distinct"]
                and independence_witness["mechanisms_distinct"]
                and independence_witness["confounder_sets_disjoint"]
            ),
            "independence_witness": independence_witness,
            "basis": "MAX_RELATIVE_DELTA",
            "tolerance_fraction": tolerance,
            "observed_max_relative_delta": observed,
            "passed": True,
            "readout_ids": [readout["readout_id"] for readout in readouts],
        },
    }


def _rederive_decision(
    run: Mapping[str, Any],
    quantity: Mapping[str, Any],
    row: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, Any]], bytes],
) -> None:
    label = f"retained decision {quantity['quantity_id']}"
    raw = artifact_loader(row["decision_artifact"])
    try:
        decision = immutable.strict_json_loads(raw, label=label)
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    decision = _exact_object(decision, label)
    _require_keys(
        decision,
        (
            "schema_version",
            "artifact_kind",
            "run_model_seal_sha256",
            "quantity_id",
            "readout_producer_sha256s",
            "outcome",
            "ncu_artifact_ids",
            "simulator_artifact_ids",
        ),
        label,
    )
    if (
        decision["schema_version"] != 1
        or decision["artifact_kind"] != "M5_QUANTITY_DECISION"
        or decision["run_model_seal_sha256"] != run["model_seal_sha256"]
        or decision["quantity_id"] != quantity["quantity_id"]
        or decision["readout_producer_sha256s"]
        != [readout["producer_artifact"]["sha256"] for readout in row["readouts"]]
        or decision["ncu_artifact_ids"] != []
        or decision["simulator_artifact_ids"] != []
    ):
        _fail(f"{label} is stale or violates Rule-1")
    declared = _normalize_outcome(decision["outcome"], f"{label}.outcome")
    if set(declared) != {"kind"} or declared["kind"] not in OUTCOME_KINDS:
        _fail(f"{label} may select only one preregistered outcome kind")
    expected = _outcome_from_kind(
        quantity, row["readouts"], row["evidence"], declared["kind"]
    )
    if row["outcome"] != expected:
        _fail(f"RESULT outcome {quantity['quantity_id']} is not rederived from retained decision bytes")


def _rederive_cross_check(
    run: Mapping[str, Any],
    quantity_id: str,
    cross_check: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, Any]], bytes],
    *,
    candidate_model_seal_sha256: str | None = None,
) -> None:
    raw = artifact_loader(cross_check["verdict_artifact"])
    try:
        verdict = immutable.strict_json_loads(raw, label="retained NCU verdict")
    except immutable.ImmutableArtifactError as exc:
        _fail(str(exc))
    verdict = _exact_object(verdict, "retained NCU verdict")
    _require_keys(
        verdict,
        (
            "schema_version",
            "artifact_kind",
            "run_model_seal_sha256",
            "quantity_id",
            "cross_check_id",
            "ncu_artifact_sha256",
            "candidate_model_seal_sha256",
            "verdict",
        ),
        "retained NCU verdict",
    )
    if candidate_model_seal_sha256 is None:
        _fail("NCU verdict validation requires the pre-NCU candidate seal")
    expected = {
        "schema_version": 1,
        "artifact_kind": "M5_NCU_VERDICT",
        "run_model_seal_sha256": run["model_seal_sha256"],
        "quantity_id": quantity_id,
        "cross_check_id": cross_check["cross_check_id"],
        "ncu_artifact_sha256": cross_check["artifact"]["sha256"],
        "candidate_model_seal_sha256": candidate_model_seal_sha256,
        "verdict": cross_check["verdict"],
    }
    if verdict != expected:
        _fail(f"NCU verdict {cross_check['cross_check_id']} is not rederived from retained bytes")


def _expected_authenticity_usages(result: Mapping[str, Any]) -> set[tuple[str, str, tuple[Any, ...]]]:
    usages: list[tuple[str, str, Mapping[str, Any]]] = []
    for row in result["quantity_results"]:
        quantity_id = row["quantity_id"]
        for role in EVIDENCE_ROLES:
            usages.extend(
                (f"EVIDENCE_{role.upper()}", f"{quantity_id}.{role}.{index}", artifact)
                for index, artifact in enumerate(row["evidence"][role])
            )
        usages.extend(
            ("RAW_SAMPLE_SERIES", readout["readout_id"], readout["raw_sample_artifact"])
            for readout in row["readouts"]
        )
        usages.extend(
            ("READOUT_PRODUCER", readout["readout_id"], readout["producer_artifact"])
            for readout in row["readouts"]
        )
        usages.append(("QUANTITY_DECISION", quantity_id, row["decision_artifact"]))
        for cross in row["cross_checks"]:
            usages.append(("NCU_RAW", cross["cross_check_id"], cross["artifact"]))
            usages.append(("NCU_VERDICT", cross["cross_check_id"], cross["verdict_artifact"]))
    protocol = result["components"]["protocol_state"]
    usages.append(("PROTOCOL_STATE_WITNESS", "protocol-state", protocol["witness_artifact"]))
    controls = result["components"]["roster_control_sidecars"]
    usages.append(("CONTROL_PC_JOIN_WITNESS", "CANONICAL_ROSTER", controls["canonical_pc_join_witness"]))
    usages.append(("CONTROL_PC_JOIN_WITNESS", "CALIBRATION_BINARIES", controls["calibration_pc_join_witness"]))
    usages.extend(
        ("CONTROL_SIDECAR", entry["cell_id"], entry["sidecar"])
        for entry in controls["entries"]
    )
    expected = {(role, context, _artifact_signature(reference)) for role, context, reference in usages}
    if len(expected) != len(usages):
        _fail("RESULT reuses one authenticated role/context/artifact binding")
    return expected


def _validate_control_bytes(
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, Any]], bytes],
) -> None:
    controls = result["components"]["roster_control_sidecars"]
    canonical_members = run["components"]["canonical_runnable_roster"]["members"]
    calibration_members = run["components"]["roster_control_sidecars"]["calibration_members"]
    canonical_cells = _parse_pc_join_witness(
        artifact_loader(controls["canonical_pc_join_witness"]),
        "retained canonical PC-join witness",
        run=run,
        expected_members=canonical_members,
    )
    calibration_cells = _parse_pc_join_witness(
        artifact_loader(controls["calibration_pc_join_witness"]),
        "retained calibration PC-join witness",
        run=run,
        expected_members=calibration_members,
    )
    members = canonical_members + calibration_members
    cells = canonical_cells | calibration_cells
    expected_entries: list[dict[str, Any]] = []
    for member, entry in zip(members, controls["entries"], strict=True):
        facts = _parse_sidecar(
            artifact_loader(entry["sidecar"]), f"retained standard sidecar {member['cell_id']}"
        )
        expected = _derived_control_entry(
            member,
            "CANONICAL_ROSTER" if member in canonical_members else "CALIBRATION_BINARIES",
            entry["sidecar"],
            entry["pc_join_witness"],
            facts,
            cells[member["cell_id"]],
        )
        expected_entries.append(expected)
        if entry != expected:
            _fail(f"control result for {member['cell_id']} is not rederived from retained bytes")
    inventory_unsigned = {
        "schema_version": 1,
        "artifact_kind": "M5_CONTROL_SIDECAR_INVENTORY",
        "run_model_seal_sha256": run["model_seal_sha256"],
        "parser_contract": copy.deepcopy(CONTROL_PARSER_CONTRACT),
        "canonical_pc_join_witness": controls["canonical_pc_join_witness"],
        "calibration_pc_join_witness": controls["calibration_pc_join_witness"],
        "entries": [
            {
                "coverage_class": entry["coverage_class"],
                "cell_id": entry["cell_id"],
                "status": entry["status"],
                "decoded_instruction_count": entry["decoded_instruction_count"],
                "unknown_encoding_count": entry["unknown_encoding_count"],
                "binary_sha256": entry["binary_sha256"],
                "pc_join_passed": entry["pc_join_passed"],
                "function_join_passed": entry["function_join_passed"],
                "sidecar_artifact_id": entry["sidecar"]["artifact_id"],
                "sidecar_sha256": entry["sidecar"]["sha256"],
                **({"typed_reasons": entry["typed_reasons"]} if entry["status"] == "TYPED" else {}),
            }
            for entry in expected_entries
        ],
    }
    expected_inventory = immutable.pretty_json_bytes(immutable.seal_model(inventory_unsigned))
    if artifact_loader(controls["inventory_artifact"]) != expected_inventory:
        _fail("control inventory is not rederived from retained standard producer bytes")


def _validate_result_semantics(
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    schema: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, Any]], bytes],
    *,
    expected_public_key_sha256: str | None = None,
) -> None:
    _verify_sealed_model(result, "M5_RESULT", schema)
    if result["run_model_seal_sha256"] != run["model_seal_sha256"]:
        _fail("RESULT does not bind exact RUN seal")
    if result["source_commit"] != run["source_state"]["source_commit"]:
        _fail("RESULT source commit differs from RUN")
    if tuple(result["family_execution_order"]) != tuple(run["family_execution_order"]) or tuple(
        result["family_scientific_rank"]
    ) != tuple(run["family_scientific_rank"]):
        _fail("RESULT family order/rank mismatch with RUN")
    _verify_authenticity_ledger(
        run,
        result,
        artifact_loader,
        expected_public_key_sha256=expected_public_key_sha256,
    )
    observed_usages = {
        (
            entry["artifact_role"],
            entry["context_id"],
            _artifact_signature(entry["artifact"]),
        )
        for entry in result["authenticity"]["entries"]
    }
    if observed_usages != _expected_authenticity_usages(result):
        _fail("RESULT authenticity receipts do not close every exact external input usage")
    ncu_entries = [
        entry
        for entry in result["authenticity"]["entries"]
        if entry["artifact_role"] in {"NCU_RAW", "NCU_VERDICT"}
    ]
    feature_entries = [
        entry
        for entry in result["authenticity"]["entries"]
        if entry["artifact_role"] not in {"NCU_RAW", "NCU_VERDICT"}
    ]
    if (
        {entry["artifact"]["artifact_id"] for entry in ncu_entries}
        & {entry["artifact"]["artifact_id"] for entry in feature_entries}
        or {entry["artifact"]["sha256"] for entry in ncu_entries}
        & {entry["artifact"]["sha256"] for entry in feature_entries}
    ):
        _fail("RESULT violates global NCU/feature byte disjointness")

    planned = run["quantities"]
    observed = result["quantity_results"]
    if len(planned) != len(observed):
        _fail("RESULT quantity count differs from RUN")
    repetition_ids: list[str] = []
    decision_hashes_by_family: dict[str, list[str]] = defaultdict(list)
    for index, (quantity, row) in enumerate(zip(planned, observed, strict=True)):
        for key in (
            "quantity_id",
            "quantity_group",
            "meaning",
            "producer",
            "consumer_binding",
            "conditioning",
            "unit",
        ):
            if row[key] != quantity[key]:
                _fail(f"RESULT quantity[{index}] changed RUN-bound {key}")
        _validate_readouts(quantity, row)
        producer_family = (
            quantity["producer"]["family_id"]
            if quantity["producer"]["kind"] == "PROBE_FAMILY"
            else "pchase"
        )
        upstream_decision_sha256s = _expected_upstream_decision_sha256s(
            decision_hashes_by_family, quantity
        )
        for contract, readout in zip(quantity["readout_contracts"], row["readouts"], strict=True):
            _rederive_readout(
                run,
                quantity,
                contract,
                readout,
                artifact_loader,
                upstream_decision_sha256s=upstream_decision_sha256s,
            )
            repetition_ids.extend(readout["repetition_ids"])
        _rederive_decision(run, quantity, row, artifact_loader)
        decision_hashes_by_family[producer_family].append(row["decision_artifact"]["sha256"])
        _validate_outcome(quantity, row)
        for role in EVIDENCE_ROLES:
            if any(artifact["size_bytes"] < 1 for artifact in row["evidence"][role]):
                _fail(f"RESULT quantity {row['quantity_id']} has empty {role} evidence")
        for cross_check in row["cross_checks"]:
            if cross_check["artifact"]["size_bytes"] < 1 or cross_check["verdict_artifact"]["size_bytes"] < 1:
                _fail(f"RESULT quantity {row['quantity_id']} has an empty NCU artifact")
            if cross_check["instrument"] != "NCU" or any(
                cross_check[field] is not False
                for field in ("fitted", "derived", "selected", "tuned", "transcribed")
            ):
                _fail(f"RESULT quantity {row['quantity_id']} violates NCU verdict-only policy")
            _rederive_cross_check(
                run,
                row["quantity_id"],
                cross_check,
                artifact_loader,
                candidate_model_seal_sha256=result["components"]["pre_ncu_candidate"]["candidate_model_seal_sha256"],
            )
        if len({cross["cross_check_id"] for cross in row["cross_checks"]}) != len(row["cross_checks"]):
            _fail(f"RESULT quantity {row['quantity_id']} has duplicate cross-check IDs")

    protocol = result["components"]["protocol_state"]
    capture_observations = _capture_observation_index(
        observed, run["family_execution_order"]
    )
    expected_protocol = _validate_protocol_state_witness(
        artifact_loader(protocol["witness_artifact"]),
        run=run,
        capture_observations=capture_observations,
        reference=protocol["witness_artifact"],
    )
    if protocol != expected_protocol:
        _fail("RESULT protocol component is not rederived from retained state witness bytes")

    pchase_ids = run["components"]["pchase"]["quantity_ids"]
    pchase = result["components"]["pchase"]
    expected_pchase_evidence = sorted(
        {
            artifact["artifact_id"]
            for row in observed
            if row["quantity_id"] in pchase_ids
            for role in EVIDENCE_ROLES
            for artifact in row["evidence"][role]
        }
    )
    if (
        pchase["component_id"] != "pchase"
        or pchase["quantity_ids"] != pchase_ids
        or pchase["evidence_artifact_ids"] != expected_pchase_evidence
        or pchase["property_count"] != 14
        or pchase["historical_card_identity_id"] != "m5-pchase-v1-card"
        or pchase["architecture_card_transcription"] is not False
    ):
        _fail("RESULT pchase inventory/lifecycle mismatch")
    temporary: dict[str, immutable.ImmutableFile] = {}
    _card, expected_card_reference = _fresh_pchase_card(run, observed, protocol, temporary)
    if pchase["fresh_card_artifact"] != expected_card_reference:
        _fail("RESULT fresh pchase card reference is stale")
    expected_card_bytes = _material_bytes(temporary[expected_card_reference["relative_path"]])
    if artifact_loader(pchase["fresh_card_artifact"]) != expected_card_bytes:
        _fail("RESULT fresh pchase card is not rederived from all 14 retained results")

    planned_controls = run["components"]["roster_control_sidecars"]
    controls = result["components"]["roster_control_sidecars"]
    canonical_count = run["components"]["canonical_runnable_roster"]["runnable_count"]
    calibration_count = len(planned_controls["calibration_members"])
    if (
        controls["component_id"] != "roster_control_sidecars"
        or controls["parser_contract"] != CONTROL_PARSER_CONTRACT
        or controls["canonical_pc_join_witness"]["artifact_id"]
        != planned_controls["expected_canonical_pc_join_witness_id"]
        or controls["calibration_pc_join_witness"]["artifact_id"]
        != planned_controls["expected_calibration_pc_join_witness_id"]
        or controls["expected_canonical_cell_count"] != canonical_count
        or controls["observed_canonical_cell_count"] != canonical_count
        or controls["expected_calibration_binary_count"] != calibration_count
        or controls["observed_calibration_binary_count"] != calibration_count
        or len(controls["entries"]) != canonical_count + calibration_count
    ):
        _fail("RESULT canonical/calibration control accounting mismatch")
    _validate_control_bytes(run, result, artifact_loader)

    feature_authenticity = {
        "authority_identity_id": result["authenticity"]["authority_identity_id"],
        "public_key_sha256": result["authenticity"]["public_key_sha256"],
        "receipt_domain": result["authenticity"]["receipt_domain"],
        "entries": [
            copy.deepcopy(entry)
            for entry in result["authenticity"]["entries"]
            if entry["artifact_role"] not in {"NCU_RAW", "NCU_VERDICT"}
        ],
    }
    candidate_model = _pre_ncu_candidate_model(
        run,
        observed,
        protocol,
        controls,
        pchase["fresh_card_artifact"],
        feature_authenticity,
    )
    candidate_bytes = immutable.pretty_json_bytes(candidate_model)
    candidate_reference = _artifact_reference(
        "m5-pre-ncu-candidate",
        immutable.sha256_bytes(candidate_bytes),
        len(candidate_bytes),
        "application/json",
    )
    expected_candidate_component = {
        "component_id": "pre_ncu_candidate",
        "artifact": candidate_reference,
        "candidate_model_seal_sha256": candidate_model["model_seal_sha256"],
        "feature_authenticated_artifact_count": len(feature_authenticity["entries"]),
        "ncu_attachment_phase": "AFTER_IMMUTABLE_CANDIDATE_FREEZE",
        "candidate_publication": {
            "publication_identity_id": f"m5-pre-ncu-publication.{candidate_model['model_seal_sha256']}",
            "artifact": candidate_reference,
            "publication_name": f"m5-pre-ncu-{candidate_model['model_seal_sha256']}",
            "verified": True,
            "verification_method": "REHASHED_PUBLISHED_BYTES",
            "published_sha256": candidate_reference["sha256"],
            "published_size_bytes": candidate_reference["size_bytes"],
        },
    }
    if result["components"]["pre_ncu_candidate"] != expected_candidate_component:
        _fail("RESULT pre-NCU candidate reference or phase closure is stale")
    if artifact_loader(candidate_reference) != candidate_bytes:
        _fail("RESULT pre-NCU candidate is not rederived from retained feature bytes")

    measured = sum(row["outcome"]["kind"] == "MEASURED" for row in observed)
    gaps = sum(row["outcome"]["kind"] in TYPED_GAPS for row in observed)
    evidence_count = sum(len(row["evidence"][role]) for row in observed for role in EVIDENCE_ROLES)
    expected_counts = {
        "required_quantity_count": len(planned),
        "result_quantity_count": len(observed),
        "measured_quantity_count": measured,
        "typed_gap_quantity_count": gaps,
        "evidence_artifact_reference_count": evidence_count,
        "raw_sample_series_count": sum(len(row["readouts"]) for row in observed),
        "pre_ncu_authenticated_artifact_count": len(feature_authenticity["entries"]),
        "authenticated_artifact_count": len(result["authenticity"]["entries"]),
        "authority_receipt_count": len(result["authenticity"]["entries"]),
        "protocol_repetition_count": len(repetition_ids),
        "canonical_control_sidecar_count": canonical_count,
        "calibration_control_sidecar_count": calibration_count,
    }
    if result["counts"] != expected_counts or len(planned) != measured + gaps:
        _fail("RESULT count/closure identity mismatch")


def _inventory_with_bindings(
    run: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    bindings: dict[str, set[str]] = defaultdict(set)

    def add(reference: Mapping[str, Any], binding: str, label: str) -> None:
        _collect_artifact(references, reference, label)
        bindings[reference["artifact_id"]].add(binding)

    for index, identity in enumerate(run["identities"]):
        add(identity["artifact"], "IDENTITY", f"identity[{index}]")
    binding_by_role = {
        "curve": "EVIDENCE_CURVE",
        "control": "EVIDENCE_CONTROL",
        "confound": "EVIDENCE_CONFOUND",
        "composition": "EVIDENCE_COMPOSITION",
    }
    for quantity_index, row in enumerate(result["quantity_results"]):
        for role, binding in binding_by_role.items():
            for index, artifact in enumerate(row["evidence"][role]):
                add(artifact, binding, f"quantity[{quantity_index}].{role}[{index}]")
        for index, cross in enumerate(row["cross_checks"]):
            add(cross["artifact"], "NCU_CROSS_CHECK", f"cross[{index}]")
            add(cross["verdict_artifact"], "NCU_VERDICT", f"cross verdict[{index}]")
        for index, readout in enumerate(row["readouts"]):
            add(readout["raw_sample_artifact"], "RAW_SAMPLE_SERIES", f"raw sample series[{quantity_index}][{index}]")
            add(readout["producer_artifact"], "READOUT_PRODUCER", f"readout producer[{quantity_index}][{index}]")
        add(row["decision_artifact"], "QUANTITY_DECISION", f"quantity decision[{quantity_index}]")
    protocol = result["components"]["protocol_state"]
    add(protocol["witness_artifact"], "PROTOCOL_STATE_WITNESS", "protocol witness")
    pchase = result["components"]["pchase"]
    add(pchase["fresh_card_artifact"], "PCHASE_FRESH_CARD", "fresh pchase card")
    candidate = result["components"]["pre_ncu_candidate"]
    add(candidate["artifact"], "PRE_NCU_CANDIDATE", "pre-NCU candidate")
    controls = result["components"]["roster_control_sidecars"]
    add(controls["inventory_artifact"], "CONTROL_INVENTORY", "control inventory")
    add(controls["canonical_pc_join_witness"], "CONTROL_PC_JOIN_WITNESS", "canonical PC witness")
    add(controls["calibration_pc_join_witness"], "CONTROL_PC_JOIN_WITNESS", "calibration PC witness")
    for index, entry in enumerate(controls["entries"]):
        add(entry["sidecar"], "CONTROL_SIDECAR", f"control sidecar[{index}]")
    for index, entry in enumerate(result["authenticity"]["entries"]):
        add(entry["artifact"], "AUTHENTICATED_INPUT", f"authenticated input[{index}]")
        add(entry["receipt"], "AUTHORITY_RECEIPT", f"authority receipt[{index}]")
    return [
        {
            **references[artifact_id],
            "bindings": sorted(bindings[artifact_id], key=_BINDING_ORDER.__getitem__),
        }
        for artifact_id in sorted(references)
    ]


def _done_model(
    run: Mapping[str, Any], run_bytes: bytes, result: Mapping[str, Any], result_bytes: bytes
) -> dict[str, Any]:
    inventory = _inventory_with_bindings(run, result)
    controls = result["components"]["roster_control_sidecars"]
    control_pins = [
        {
            "coverage_class": entry["coverage_class"],
            "cell_id": entry["cell_id"],
            "artifact_id": entry["sidecar"]["artifact_id"],
            "relative_path": entry["sidecar"]["relative_path"],
            "sha256": entry["sidecar"]["sha256"],
            "size_bytes": entry["sidecar"]["size_bytes"],
        }
        for entry in controls["entries"]
    ]
    unsigned = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "artifact_kind": "M5_DONE",
        "source_commit": run["source_state"]["source_commit"],
        "target": copy.deepcopy(run["target"]),
        "run_manifest": {
            "artifact_kind": "M5_RUN",
            "relative_path": "RUN.json",
            "sha256": immutable.sha256_bytes(run_bytes),
            "size_bytes": len(run_bytes),
            "model_seal_sha256": run["model_seal_sha256"],
        },
        "result_manifest": {
            "artifact_kind": "M5_RESULT",
            "relative_path": "RESULT.json",
            "sha256": immutable.sha256_bytes(result_bytes),
            "size_bytes": len(result_bytes),
            "model_seal_sha256": result["model_seal_sha256"],
        },
        "artifact_inventory": inventory,
        "analysis_authority": {
            "identity_id": result["authenticity"]["authority_identity_id"],
            "public_key_sha256": result["authenticity"]["public_key_sha256"],
            "receipt_domain": result["authenticity"]["receipt_domain"],
        },
        "authenticity_receipts": [
            entry["receipt"] for entry in result["authenticity"]["entries"]
        ],
        "protocol_witness": copy.deepcopy(
            result["components"]["protocol_state"]["witness_artifact"]
        ),
        "fresh_pchase_card": copy.deepcopy(
            result["components"]["pchase"]["fresh_card_artifact"]
        ),
        "pre_ncu_candidate": copy.deepcopy(result["components"]["pre_ncu_candidate"]),
        "control_sidecars": control_pins,
        "counts": {
            "required_quantity_count": len(run["quantities"]),
            "measured_quantity_count": result["counts"]["measured_quantity_count"],
            "typed_gap_quantity_count": result["counts"]["typed_gap_quantity_count"],
            "result_quantity_count": len(result["quantity_results"]),
            "unique_artifact_count": len(inventory),
            "raw_sample_series_count": result["counts"]["raw_sample_series_count"],
            "pre_ncu_authenticated_artifact_count": result["counts"][
                "pre_ncu_authenticated_artifact_count"
            ],
            "authenticated_artifact_count": result["counts"]["authenticated_artifact_count"],
            "authority_receipt_count": result["counts"]["authority_receipt_count"],
            "canonical_control_sidecar_count": result["counts"]["canonical_control_sidecar_count"],
            "calibration_control_sidecar_count": result["counts"]["calibration_control_sidecar_count"],
            "control_sidecar_count": len(control_pins),
        },
        "closure": {
            "required_equals_measured_plus_typed_gap": True,
            "required_equals_result_count": True,
            "all_required_quantities_closed": True,
            "all_artifacts_hash_pinned": True,
            "all_authenticated_inputs_receipted": True,
            "protocol_state_join_closed": True,
            "fresh_pchase_card_closed": True,
            "raw_sample_lineage_closed": True,
            "pre_ncu_candidate_frozen_before_ncu_reads": True,
            "ncu_attachment_after_candidate_freeze": True,
            "canonical_122_control_roster_closed": True,
            "all_control_sidecars_hash_pinned": True,
        },
        "rule1_attestation": {
            "attested": True,
            "ncu_role": "CROSS_CHECK_ONLY",
            "ncu_feature_bytes_disjoint": True,
            "ncu_derived": False,
            "ncu_selected": False,
            "ncu_tuned": False,
            "ncu_transcribed": False,
            "simulator_output_used": False,
        },
        "controller_card_boundary": {
            "attested": True,
            "controller_writes_architecture_cards": False,
            "publication_path_is_architecture_card": False,
            "output_mode": "ADDITIVE_EVIDENCE_ONLY",
        },
        "publication": {
            "transaction_protocol": "O_EXCL_O_NOFOLLOW_FSYNC_RENAME_NOREPLACE_V1",
            "deterministic_stage": True,
            "no_replace_rename": True,
            "exact_idempotent_resume": True,
            "read_only_files": True,
            "read_only_directories": True,
            "relocation_safe": True,
            "verify_rehashes_bytes": True,
            "verify_uses_pinned_schemas": True,
            "partial_recovery_scope": "EXACT_COMPLETE_ONLY_PRESERVE_INCOMPLETE",
        },
    }
    return immutable.seal_model(unsigned)


def _validate_done_semantics(
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    done: Mapping[str, Any],
    run_bytes: bytes,
    result_bytes: bytes,
    schema: Mapping[str, Any],
) -> None:
    _verify_sealed_model(done, "M5_DONE", schema)
    if done["source_commit"] != run["source_state"]["source_commit"] or done["target"] != run["target"]:
        _fail("DONE source/target identity differs from RUN")
    expected_run_pin = {
        "artifact_kind": "M5_RUN",
        "relative_path": "RUN.json",
        "sha256": immutable.sha256_bytes(run_bytes),
        "size_bytes": len(run_bytes),
        "model_seal_sha256": run["model_seal_sha256"],
    }
    expected_result_pin = {
        "artifact_kind": "M5_RESULT",
        "relative_path": "RESULT.json",
        "sha256": immutable.sha256_bytes(result_bytes),
        "size_bytes": len(result_bytes),
        "model_seal_sha256": result["model_seal_sha256"],
    }
    if done["run_manifest"] != expected_run_pin or done["result_manifest"] != expected_result_pin:
        _fail("DONE manifest pins are stale")
    inventory = _inventory_with_bindings(run, result)
    if done["artifact_inventory"] != inventory:
        _fail("DONE artifact inventory is not exact RUN+RESULT closure")
    expected_authority = {
        "identity_id": result["authenticity"]["authority_identity_id"],
        "public_key_sha256": result["authenticity"]["public_key_sha256"],
        "receipt_domain": result["authenticity"]["receipt_domain"],
    }
    if done["analysis_authority"] != expected_authority:
        _fail("DONE analysis authority trust binding is stale")
    if done["authenticity_receipts"] != [
        entry["receipt"] for entry in result["authenticity"]["entries"]
    ]:
        _fail("DONE receipt inventory is stale")
    if done["protocol_witness"] != result["components"]["protocol_state"]["witness_artifact"]:
        _fail("DONE protocol witness pin is stale")
    if done["fresh_pchase_card"] != result["components"]["pchase"]["fresh_card_artifact"]:
        _fail("DONE fresh pchase card pin is stale")
    if done["pre_ncu_candidate"] != result["components"]["pre_ncu_candidate"]:
        _fail("DONE pre-NCU candidate pin or phase closure is stale")
    entries = result["components"]["roster_control_sidecars"]["entries"]
    expected_controls = [
        {
            "coverage_class": entry["coverage_class"],
            "cell_id": entry["cell_id"],
            "artifact_id": entry["sidecar"]["artifact_id"],
            "relative_path": entry["sidecar"]["relative_path"],
            "sha256": entry["sidecar"]["sha256"],
            "size_bytes": entry["sidecar"]["size_bytes"],
        }
        for entry in entries
    ]
    if done["control_sidecars"] != expected_controls:
        _fail("DONE control-sidecar pins are stale")
    expected_counts = {
        "required_quantity_count": len(run["quantities"]),
        "measured_quantity_count": result["counts"]["measured_quantity_count"],
        "typed_gap_quantity_count": result["counts"]["typed_gap_quantity_count"],
        "result_quantity_count": len(result["quantity_results"]),
        "unique_artifact_count": len(inventory),
        "raw_sample_series_count": result["counts"]["raw_sample_series_count"],
        "pre_ncu_authenticated_artifact_count": result["counts"][
            "pre_ncu_authenticated_artifact_count"
        ],
        "authenticated_artifact_count": result["counts"]["authenticated_artifact_count"],
        "authority_receipt_count": result["counts"]["authority_receipt_count"],
        "canonical_control_sidecar_count": result["counts"]["canonical_control_sidecar_count"],
        "calibration_control_sidecar_count": result["counts"]["calibration_control_sidecar_count"],
        "control_sidecar_count": len(expected_controls),
    }
    if done["counts"] != expected_counts or expected_counts["required_quantity_count"] != (
        expected_counts["measured_quantity_count"] + expected_counts["typed_gap_quantity_count"]
    ):
        _fail("DONE count/closure identity mismatch")
    if done != _done_model(run, run_bytes, result, result_bytes):
        _fail("DONE controller-derived attestations or publication contract are stale")


@_contract_boundary
def analyze(run_path: Path, request_path: Path, output_root: Path) -> Path:
    root = _external_output_root(output_root)
    request_path = Path(os.path.abspath(os.fspath(request_path)))
    _refuse_card_path(request_path, "analysis request")
    run_root, run, run_bytes, schemas, run_loader = _verify_run_bundle(run_path)
    analysis_loader: _DeclaredSnapshotLoader | None = None
    validation_loader: _DeclaredSnapshotLoader | None = None
    try:
        _require_output_disjoint_from_run(root, run_root)
        _reauthorize_run_source_and_controller(run)
        request, _ = _load_json_file(request_path, "M5 analysis request")
        result, analysis_material, analysis_loader = _analysis_request_to_result(
            request, request_path, run, run_loader
        )
        analysis_loader.freeze_into(analysis_material)
        for reference in _run_artifacts(run).values():
            run_loader(reference)
        run_material: dict[str, immutable.ImmutableFile] = {
            reference["relative_path"]: immutable.ImmutableFile.from_bytes(
                reference["relative_path"], run_loader(reference)
            )
            for reference in _run_artifacts(run).values()
        }
        all_material = dict(run_material)
        for material in analysis_material.values():
            _merge_material(all_material, material)
        run_loader.freeze_into(all_material)
        validation_loader = _DeclaredSnapshotLoader(material=all_material)
        _validate_result_semantics(run, result, schemas["M5_RESULT"], validation_loader)
        validation_loader.freeze_into(all_material)
        result_bytes = immutable.pretty_json_bytes(result)
        done = _done_model(run, run_bytes, result, result_bytes)
        _validate_done_semantics(
            run, result, done, run_bytes, result_bytes, schemas["M5_DONE"]
        )
        done_bytes = immutable.pretty_json_bytes(done)
        all_material["RUN.json"] = immutable.ImmutableFile.from_bytes("RUN.json", run_bytes)
        all_material["RESULT.json"] = immutable.ImmutableFile.from_bytes("RESULT.json", result_bytes)
        all_material["DONE.json"] = immutable.ImmutableFile.from_bytes("DONE.json", done_bytes)

        def commit_guard(commit_root: Path, descriptor: int) -> None:
            _authorize_output_descriptor(commit_root, descriptor)
            _require_output_disjoint_from_run(commit_root, run_root)
            run_loader.verify_root()
            _reauthorize_run_source_and_controller(run)

        try:
            return immutable.publish_directory(
                root,
                f"m5-calibration-{done['model_seal_sha256']}",
                all_material.values(),
                before_commit=commit_guard,
            )
        except immutable.ImmutableArtifactError as exc:
            _fail(str(exc))
    finally:
        if validation_loader is not None:
            validation_loader.close()
        if analysis_loader is not None:
            analysis_loader.close()
        run_loader.close()


@_contract_boundary
def verify(value: Path, expected_authority_sha256: str) -> dict[str, Any]:
    if type(expected_authority_sha256) is not str or re.fullmatch(
        r"[0-9a-f]{64}", expected_authority_sha256
    ) is None:
        _fail("verify requires an out-of-band lowercase SHA-256 authority trust anchor")
    root, done_relative = _bundle_path(value, "DONE.json", "DONE bundle")
    loader = _DeclaredSnapshotLoader(root=root)
    try:
        done, done_bytes = loader.read_manifest(done_relative, "M5 DONE")
        _require_pretty_bytes(done, done_bytes, "M5 DONE")
        try:
            immutable.verify_model_seal(done)
        except immutable.ImmutableArtifactError as exc:
            _fail(f"M5_DONE canonical model seal: {exc}")
        run_pin = _exact_object(done.get("run_manifest"), "DONE run_manifest")
        result_pin = _exact_object(done.get("result_manifest"), "DONE result_manifest")
        run_path = _exact_string(run_pin.get("relative_path"), "DONE run relative_path")
        result_path = _exact_string(result_pin.get("relative_path"), "DONE result relative_path")
        if run_path != "RUN.json" or result_path != "RESULT.json" or done_relative != "DONE.json":
            _fail("DONE bundle manifests do not use the exact canonical root paths")
        run, run_bytes = loader.read_manifest(run_path, "M5 RUN")
        result, result_bytes = loader.read_manifest(result_path, "M5 RESULT")
        _require_pretty_bytes(run, run_bytes, "M5 RUN")
        _require_pretty_bytes(result, result_bytes, "M5 RESULT")
        schemas = _pinned_schema_map(run, loader)
        _validate_run_semantics(run, schemas["M5_RUN"], loader)
        _validate_result_semantics(
            run,
            result,
            schemas["M5_RESULT"],
            loader,
            expected_public_key_sha256=expected_authority_sha256,
        )
        _validate_done_semantics(
            run, result, done, run_bytes, result_bytes, schemas["M5_DONE"]
        )
        expected = [
            immutable.ImmutableFile(
                done_relative,
                immutable.sha256_bytes(done_bytes),
                len(done_bytes),
                source_path=root / done_relative,
            ),
            immutable.ImmutableFile(
                run_path,
                run_pin["sha256"],
                run_pin["size_bytes"],
                source_path=root / run_path,
            ),
            immutable.ImmutableFile(
                result_path,
                result_pin["sha256"],
                result_pin["size_bytes"],
                source_path=root / result_path,
            ),
        ]
        expected.extend(
            immutable.ImmutableFile(
                row["relative_path"],
                row["sha256"],
                row["size_bytes"],
                source_path=root / row["relative_path"],
            )
            for row in done["artifact_inventory"]
        )
        try:
            immutable.verify_published_files(root, expected)
        except immutable.ImmutableArtifactError as exc:
            _fail(f"relocated bundle byte verification failed: {exc}")
        loader.verify_root()
        return done
    finally:
        loader.close()


@_contract_boundary
def collect(*_arguments: Any, **_keywords: Any) -> None:
    _fail(M5_COLLECT_DISABLED)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan_parser = subcommands.add_parser("plan")
    plan_parser.add_argument("--request", "--input", dest="request", type=Path, required=True)
    plan_parser.add_argument("--output-root", "--output", dest="output_root", type=Path, required=True)
    analyze_parser = subcommands.add_parser("analyze")
    analyze_parser.add_argument("--run", type=Path, required=True)
    analyze_parser.add_argument("--request", "--input", dest="request", type=Path, required=True)
    analyze_parser.add_argument("--output-root", "--output", dest="output_root", type=Path, required=True)
    verify_parser = subcommands.add_parser("verify")
    verify_parser.add_argument("bundle_positional", nargs="?", type=Path)
    verify_parser.add_argument("--bundle", dest="bundle_option", type=Path)
    verify_parser.add_argument("--expected-authority-sha256", required=True)
    collect_parser = subcommands.add_parser("collect")
    collect_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "collect":
        sys.stderr.write(f"{M5_COLLECT_DISABLED}\n")
        return 2
    parser = _parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "plan":
            print(plan(args.request, args.output_root))
            return 0
        if args.command == "analyze":
            print(analyze(args.run, args.request, args.output_root))
            return 0
        if args.command == "verify":
            bundle = args.bundle_option or args.bundle_positional
            if bundle is None or (
                args.bundle_option is not None and args.bundle_positional is not None
            ):
                parser.error("verify requires exactly one bundle path")
            done = verify(bundle, args.expected_authority_sha256)
            print(f"M5_VERIFIED {done['model_seal_sha256']}")
            return 0
        _fail(f"unsupported command: {args.command}")
    except (M5ContractError, immutable.ImmutableArtifactError) as exc:
        sys.stderr.write(f"M5_CONTRACT_ERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
