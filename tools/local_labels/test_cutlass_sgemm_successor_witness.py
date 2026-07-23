from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.local_labels import build_cutlass_sgemm_successor_witness as witness


REPO = Path(__file__).resolve().parents[2]
WITNESS_PATH = REPO / witness.OUT_REL


def _load() -> dict:
    assert WITNESS_PATH.is_file(), "tracked FP32 successor witness is missing"
    return json.loads(WITNESS_PATH.read_text())


def test_fp32_successor_witness_is_sealed_and_valid():
    document = _load()
    assert witness.validate_witness(document) == []
    assert document["pass"] is True
    assert document["witness_id"] == witness.WITNESS_ID
    assert document["base_frozen_v1"]["base_slug"] == "vendor_sgemm_cublas"


def test_fp32_successor_has_independent_direct_control_and_shape_contract():
    document = _load()
    assert document["artifacts"]["control_bits"]["unknown_encoding"] == 0
    assert document["control_contract"]["policy"] == "direct_executable"
    assert document["shape_authority"]["slug"] == "vendor_sgemm_cutlass"
    assert document["shape_authority"]["shape_role"] == "trace"
    assert document["smoke"]["reported_shape_point"] == "trace_00"
    assert all("/tmp/" not in str(value) for value in document["smoke"]["command"])
    assert all("/tmp/" not in str(value) for value in document["artifacts"]["control_bits"]["command"])


def test_fp32_successor_rejects_resealed_shape_and_control_drift():
    document = _load()
    mutated = copy.deepcopy(document)
    mutated["smoke"]["reported_shape_point"] = "timing_00"
    assert witness.validate_witness(mutated)

    mutated = copy.deepcopy(document)
    mutated["artifacts"]["control_bits"]["unknown_encoding"] = 1
    assert witness.validate_witness(mutated)


def test_fp32_shape_arguments_are_explicit_and_independent():
    shape, args, digest = witness._load_shape()
    assert shape["name"] == "trace_00"
    assert shape["axes"] == {"m": 32, "n": 32, "k": 32}
    assert args == [
        "--m", "32", "--n", "32", "--k", "32", "--seed", "42",
        "--shape-point", "trace_00",
    ]
    assert len(digest) == 64
