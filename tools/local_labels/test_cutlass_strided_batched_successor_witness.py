from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from tools.local_labels import (
    build_cutlass_strided_batched_successor_witness as witness,
)


REPO = Path(__file__).resolve().parents[2]
WITNESS_PATH = REPO / witness.OUT_REL


def _load() -> dict:
    assert WITNESS_PATH.is_file(), "tracked strided-batched successor witness is missing"
    return json.loads(WITNESS_PATH.read_text())


def _reseal(document: dict) -> None:
    document["witness_seal_sha256"] = witness._canonical_sha(
        {
            key: value
            for key, value in document.items()
            if key != "witness_seal_sha256"
        }
    )


def test_strided_batched_successor_witness_is_sealed_and_valid():
    document = _load()
    assert witness.validate_witness(document) == []
    assert document["pass"] is True
    assert document["witness_id"] == witness.WITNESS_ID
    assert document["source_commit"] == "f5837b0c437d20b5f2573903d435c0b4c6819b58"
    assert (
        document["base_frozen_v1"]["base_authority_cell_sha256"]
        == witness.FROZEN_BASE_CELL_SHA256
    )
    assert (
        "tools/local_labels/build_cutlass_successor_witness.py"
        in document["source_inputs"]["files"]
    )


def test_strided_batched_successor_has_direct_control_and_bounded_output():
    document = _load()
    control = document["artifacts"]["control_bits"]
    assert control["unknown_encoding"] == 0
    assert control["toolkit_version"] == "V13.2"
    assert document["control_contract"]["policy"] == "direct_executable"
    assert document["control_contract"]["hardware_capture_status"] == (
        "not_run_build_witness_only"
    )
    assert document["shape_authority"]["shape_contract"] == (
        "trace_shape_is_actual_direct_executable_launch_with_load_bearing_batch_count"
    )
    assert document["shape_authority"]["output_scope"] == "batch_zero_only"
    assert document["smoke"]["reported_output_scope"] == "batch_zero_only"
    assert document["smoke"]["reported_batch_count"] == 4
    assert document["artifacts"]["no_cublas"]["pass"] is True
    assert all("/tmp/" not in str(value) for value in document["smoke"]["command"])
    assert all(
        "/tmp/" not in str(value)
        for value in document["artifacts"]["control_bits"]["command"]
    )


def test_strided_batched_shape_arguments_are_explicit_and_load_bearing():
    shape, args, digest = witness._load_shape()
    assert shape["name"] == "trace_00"
    assert shape["axes"] == {"m": 32, "n": 32, "k": 32, "batch_count": 4}
    assert shape["launch_config"] == {
        "grid": [2, 2, 4],
        "block": [64, 1, 1],
    }
    assert shape["predicted_bytes"] == 20971520
    assert args == [
        "--m",
        "32",
        "--n",
        "32",
        "--k",
        "32",
        "--batch-count",
        "4",
        "--seed",
        "42",
        "--shape-point",
        "trace_00",
    ]
    assert len(digest) == 64


def test_strided_batched_successor_rejects_resealed_contract_drift():
    document = _load()

    mutated = copy.deepcopy(document)
    mutated["smoke"]["reported_batch_count"] = 1
    _reseal(mutated)
    assert witness.validate_witness(mutated)

    mutated = copy.deepcopy(document)
    mutated["shape_authority"]["output_scope"] = "full_batch"
    mutated["smoke"]["reported_output_scope"] = "full_batch"
    _reseal(mutated)
    assert witness.validate_witness(mutated)

    mutated = copy.deepcopy(document)
    mutated["artifacts"]["control_bits"]["unknown_encoding"] = 1
    _reseal(mutated)
    assert witness.validate_witness(mutated)


def test_strided_batched_sass_is_pinned_simt_sm86_evidence():
    document = _load()
    sass_path = REPO / document["artifacts"]["sass"]["path"]
    sass = sass_path.read_text()
    assert "GemmBatchedIdentityThreadblockSwizzle" in sass
    assert len(re.findall(r"\bFFMA\b", sass)) == 576
    assert re.search(r"\bHMMA\b", sass) is None
    assert document["artifacts"]["cubin_inventory"]["sm86_cubin_present"] is True
