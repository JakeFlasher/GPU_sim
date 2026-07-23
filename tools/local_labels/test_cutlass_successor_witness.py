from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.local_labels import build_cutlass_successor_witness as witness


REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / witness.DEFAULT_EVIDENCE_REL
WITNESS_PATH = EVIDENCE / "vendor_gemm_cutlass.build_witness.json"


def _load() -> dict:
    return json.loads(WITNESS_PATH.read_text())


def test_committed_successor_witness_is_schema_and_hash_valid():
    assert WITNESS_PATH.is_file(), "tracked BF16 successor witness is missing"
    document = _load()
    assert witness.validate_witness(document, repo_root=REPO) == []
    assert document["pass"] is True
    assert document["arch"] == "sm_86"
    assert document["base_frozen_v1"]["claim_policy"] == (
        "none_additive_supplement_only"
    )


def test_successor_witness_keeps_control_contract_direct_and_unknown_free():
    assert WITNESS_PATH.is_file(), "tracked BF16 successor witness is missing"
    document = _load()
    control = document["artifacts"]["control_bits"]
    assert document["control_contract"]["policy"] == "direct_executable"
    assert control["toolkit_version"] == "V13.2"
    assert control["unknown_encoding"] == 0
    assert control["decoded_instructions"] > 0
    assert all("/tmp/" not in str(value) for value in control["command"])
    assert all("/tmp/" not in str(value) for value in document["smoke"]["command"])


def test_successor_witness_rejects_shape_or_artifact_drift():
    assert WITNESS_PATH.is_file(), "tracked BF16 successor witness is missing"
    document = _load()
    mutated = copy.deepcopy(document)
    mutated["smoke"]["reported_shape_point"] = "correctness_00"
    assert witness.validate_witness(mutated, repo_root=REPO)

    mutated = copy.deepcopy(document)
    mutated["artifacts"]["control_bits"]["unknown_encoding"] = 1
    assert witness.validate_witness(mutated, repo_root=REPO)


def test_successor_shape_contract_is_actual_trace_launch():
    shape, args, shape_digest = witness.load_shape()
    assert shape["name"] == "trace_00"
    assert shape["axes"] == {"m": 32, "n": 32, "k": 32}
    assert args == [
        "--m", "32", "--n", "32", "--k", "32", "--seed", "42",
        "--shape-point", "trace_00",
    ]
    assert len(shape_digest) == 64


def test_frozen_base_pin_matches_immutable_authority():
    base = witness.load_frozen_base()
    assert base["matrix_file_sha256"] == witness.FROZEN_MATRIX_SHA256
    assert base["matrix_seal_sha256"] == witness.FROZEN_MATRIX_SEAL
    assert base["matrix_source_commit"] == witness.FROZEN_SOURCE_COMMIT
    assert base["base_authority_cell_sha256"] == witness.FROZEN_BASE_CELL_SHA256
    assert base["relation"] == "evaluation_successor_no_relabel"
