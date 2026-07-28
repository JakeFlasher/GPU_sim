"""Closed-schema sidecar: per-method enforcement + inadmissible-method rejection."""
import hashlib

import pytest

from counting_pass import BlockStat, CellStats
from sidecar import (SidecarSemanticError, build_sidecar, is_valid,
                     sidecar_sha256, validate_sidecar, write_sidecar)


class _Comp:
    raw_bytes = 1000
    compressed_bytes = 200
    decompressed_sha256 = "a" * 64


class _Gate:
    def __init__(self, *, certainty_bytes=None, max_trace_bytes=None,
                 certainty_budget_exceeded=False, coop=False,
                 verdict="subset_admissible", skew_heavy_block_count=1):
        self.certainty_bytes = certainty_bytes
        self.max_trace_bytes = max_trace_bytes
        self.certainty_budget_exceeded = certainty_budget_exceeded
        self.coop = coop
        self.verdict = verdict
        self.skew_heavy_block_count = skew_heavy_block_count

    def to_sidecar_gates(self):
        return {
            "coop": self.coop,
            "wave_count": 60,
            "n_measured_waves": 40,
            "tail_measured_waves": 2,
            "atomic_density": 0.0,
            "hotline_share": 0.0,
            "skew_top1pct_share": 0.0,
            "gini": 0.0,
            "per_sm_block_count_cv": 0.0,
            "n_phases": 1,
            "skew_heavy_block_count": self.skew_heavy_block_count,
            "certainty_bytes": self.certainty_bytes,
            "max_trace_bytes": self.max_trace_bytes,
            "certainty_budget_exceeded": self.certainty_budget_exceeded,
            "verdict": self.verdict,
        }


def _stats(n_waves=60):
    blocks = [BlockStat(block_id=i, sm_id=0, records=100)
              for i in range(60)]
    for block in blocks:
        block.wave = block.block_id
    return CellStats(
        cell_id="c", kernel_name="_Z1kv", grid=(60, 1, 1), block=(32, 1, 1),
        n_blocks=60, total_records=6000, op_class_totals={1: 6000}, blocks=blocks,
        per_sm_resident={0: 1}, per_sm_block_ids={0: list(range(60))},
        n_waves=n_waves,
        atomic_records=0, atomic_top256_sector_share=0.0,
        per_wave_record_totals={wave: 100 for wave in range(60)},
        per_sm_block_counts={0: 60})


def _compressed(**over):
    s = _stats()
    sc = build_sidecar(
        cell_id="c", kernel_name="_Z1kv", args_hash="deadbeef", stats=s,
        gate=_Gate(), compression=_Comp(), records_captured=200,
        tracer_version=5, stats_pass_version=1,
        method="lossless_compressed_full", reason="byte_budget")
    sc.update(over)
    return sc


def _subset():
    s = _stats()
    return build_sidecar(
        cell_id="c", kernel_name="_Z1kv", args_hash="ab", stats=s,
        gate=_Gate(certainty_bytes=100, max_trace_bytes=2000),
        compression=_Comp(), records_captured=200, tracer_version=5,
        stats_pass_version=2, method="wave_segment_stratified_subset",
        reason="byte_budget", rng_algo="pcg64",
        segments=[{"id": 0, "block_ids": [0, 1, 2]},
                  {"id": 19, "block_ids": [57, 58, 59]},
                  {"id": 1, "block_ids": [3, 4, 5]},
                  {"id": 2, "block_ids": [6, 7, 8]}],
        selected_segment_ids=[0, 19, 1, 2],
        certainty_segments=[0, 19],
        force_include_blocks=[0],
        strata=[{"id": 0, "phase_id": 0, "launch_tercile": 0,
                 "segment_ids": [1, 2], "N_h": 6, "n_h": 2, "pi": 1 / 3,
                 "sector_stddev": 2.0, "neyman_weight": 12.0}],
        selection_reconciliation={
            "status": "fixed_point", "attempts": 1, "max_attempts": 3,
            "capture_stats_sha256": "b" * 64,
        },
        selection_budget={
            "max_trace_bytes": 2000, "basis": "raw_threadblock_pb",
            "certainty_fraction_numerator": 2,
            "certainty_fraction_denominator": 5,
            "certainty_bytes": 100,
        },
        selection_auxiliary={
            "name": "l2_sector_transactions_32b", "sector_bytes": 32,
            "scope": "pass2_all_blocks", "expected_full": 400,
            "selected": 200,
        },
        estimator={"mode": "weighted_full_shape_estimate", "targets": ["sectors"],
                   "not_recoverable": ["cycles"],
                   "convergence": {"status": "converged"}})


def test_compressed_sidecar_is_valid_and_labels_full_shape():
    sc = _compressed()
    validate_sidecar(sc)
    assert sc["label_pairing"] == "full_shape_label"
    assert sc["hashes"]["decompressed_payload_sha256"] == "a" * 64


def test_compressed_without_payload_hash_is_rejected():
    sc = _compressed()
    del sc["hashes"]["decompressed_payload_sha256"]
    assert not is_valid(sc)  # the compression contract hash is required


def test_subset_without_payload_hash_is_rejected():
    sc = _subset()
    assert sc["hashes"]["decompressed_payload_sha256"] == "a" * 64
    del sc["hashes"]["decompressed_payload_sha256"]
    assert not is_valid(sc)


def test_subset_requires_a_seal_and_canonical_payload_hash(tmp_path):
    sc = _subset()
    del sc["hashes"]["sidecar_sha256"]
    assert not is_valid(sc)
    with pytest.raises(Exception):
        write_sidecar(sc, tmp_path / "unsigned.json")

    sc = _subset()
    sc["hashes"]["decompressed_payload_sha256"] = "x"
    assert not is_valid(sc)


def test_sidecar_write_never_overwrites_existing_evidence(tmp_path):
    path = tmp_path / "trace_sampling.json"
    path.write_text("preserve me\n")
    with pytest.raises(FileExistsError):
        write_sidecar(_subset(), path)
    assert path.read_text() == "preserve me\n"


def test_extra_key_is_rejected():
    assert not is_valid(_compressed(surprise=1))


def test_inadmissible_methods_are_rejected():
    assert not is_valid(_compressed(method="per_record_reservoir"))
    assert not is_valid(_compressed(method="deterministic_every_k"))


def test_subset_sidecar_with_all_required_parts_is_valid():
    sc = _subset()
    validate_sidecar(sc)
    assert sc["schema_version"] == 2
    assert sc["segments"][0]["block_ids"] == [0, 1, 2]
    assert sc["label_pairing"] == "full_shape_label"
    assert "wave_model" in sc and "per_sm_slot_map" in sc
    assert sc["selection_reconciliation"]["status"] == "fixed_point"
    assert sc["selection_budget"]["basis"] == "raw_threadblock_pb"
    assert sc["selection_auxiliary"]["scope"] == "pass2_all_blocks"
    assert sc["strata"][0]["phase_id"] == 0
    assert sc["strata"][0]["launch_tercile"] == 0
    assert sc["wave_model"]["per_sm_resident_blocks"] == [
        {"sm_id": 0, "resident_blocks": 1}]


def test_sealed_sidecar_rejects_stale_hash_after_mutation():
    sc = _subset()
    sc["records"]["captured"] += 1
    with pytest.raises(SidecarSemanticError, match="sidecar_sha256"):
        validate_sidecar(sc)


def test_subset_missing_estimator_is_rejected():
    # Per-method if/then: a subset sidecar without segments/strata/estimator fails.
    sc = _subset()
    del sc["estimator"]
    assert not is_valid(sc)
    with pytest.raises(Exception):
        validate_sidecar(sc)


def test_subset_missing_segments_is_rejected():
    sc = _subset()
    del sc["segments"]
    assert not is_valid(sc)


@pytest.mark.parametrize("field", [
    "selected_segment_ids", "certainty_segments", "force_include_blocks",
    "selection_reconciliation", "selection_budget", "selection_auxiliary",
])
def test_subset_missing_binding_evidence_is_rejected(field):
    sc = _subset()
    del sc[field]
    assert not is_valid(sc)


@pytest.mark.parametrize(("container", "field"), [
    ("selection_reconciliation", "status"),
    ("selection_reconciliation", "attempts"),
    ("selection_reconciliation", "max_attempts"),
    ("selection_reconciliation", "capture_stats_sha256"),
    ("selection_budget", "max_trace_bytes"),
    ("selection_budget", "basis"),
    ("selection_budget", "certainty_fraction_numerator"),
    ("selection_budget", "certainty_fraction_denominator"),
    ("selection_budget", "certainty_bytes"),
    ("selection_auxiliary", "name"),
    ("selection_auxiliary", "sector_bytes"),
    ("selection_auxiliary", "scope"),
    ("selection_auxiliary", "expected_full"),
    ("selection_auxiliary", "selected"),
])
def test_subset_binding_envelopes_are_closed_and_required(container, field):
    sc = _subset()
    del sc[container][field]
    assert not is_valid(sc)


@pytest.mark.parametrize("field", [
    "phase_id", "launch_tercile", "sector_stddev", "neyman_weight",
])
def test_subset_strata_require_design_provenance(field):
    sc = _subset()
    del sc["strata"][0][field]
    assert not is_valid(sc)


def test_subset_stratum_id_encodes_phase_and_launch_tercile():
    sc = _subset()
    sc["strata"][0]["id"] = 1
    with pytest.raises(SidecarSemanticError, match="does not encode"):
        validate_sidecar(sc)


def test_subset_selected_membership_must_match_segments_and_strata():
    sc = _subset()
    sc["selected_segment_ids"] = [0, 1]
    with pytest.raises(SidecarSemanticError, match="segments ids"):
        validate_sidecar(sc)

    sc = _subset()
    sc["strata"][0]["segment_ids"] = [1, 3]
    with pytest.raises(SidecarSemanticError, match="selected_segment_ids"):
        validate_sidecar(sc)


def test_subset_stratum_probability_and_neyman_weight_are_exact():
    sc = _subset()
    sc["strata"][0]["pi"] = 0.5
    with pytest.raises(SidecarSemanticError, match="pi"):
        validate_sidecar(sc)

    sc = _subset()
    sc["strata"][0]["neyman_weight"] = 3.9
    with pytest.raises(SidecarSemanticError, match="neyman_weight"):
        validate_sidecar(sc)


def test_subset_stratum_probability_has_no_absolute_tolerance_floor():
    sc = _subset()
    sc["strata"][0].update({
        "N_h": (1 << 64) - 1,
        "pi": 1e-13,
        "sector_stddev": 0.0,
        "neyman_weight": 0.0,
    })
    sc["hashes"].pop("sidecar_sha256")
    sc["hashes"]["sidecar_sha256"] = sidecar_sha256(sc)

    with pytest.raises(SidecarSemanticError, match="pi"):
        validate_sidecar(sc)


@pytest.mark.parametrize(("field", "value"), [
    ("pi", float("nan")),
    ("sector_stddev", float("inf")),
    ("neyman_weight", float("inf")),
])
def test_subset_stratum_design_numbers_must_be_finite(field, value):
    sc = _subset()
    sc["strata"][0][field] = value
    with pytest.raises(SidecarSemanticError, match=f"{field} must be finite"):
        validate_sidecar(sc)


def test_subset_reconciliation_attempts_are_bounded():
    sc = _subset()
    sc["selection_reconciliation"]["attempts"] = 4
    with pytest.raises(SidecarSemanticError, match="attempts"):
        validate_sidecar(sc)


def test_force_includes_must_be_inside_certainty_segments():
    sc = _subset()
    sc["force_include_blocks"] = [3]
    with pytest.raises(SidecarSemanticError, match="force_include_blocks"):
        validate_sidecar(sc)


def test_reconstructed_cold_and_tail_segments_are_always_certainty_units():
    sc = _subset()
    sc["certainty_segments"] = [0]
    with pytest.raises(SidecarSemanticError, match="cold/tail"):
        validate_sidecar(sc)


def test_subset_gate_and_force_evidence_must_be_consistent():
    sc = _subset()
    sc["gates"]["coop"] = True
    sc["gates"]["verdict"] = "coop_forbid"
    with pytest.raises(SidecarSemanticError, match="verdict"):
        validate_sidecar(sc)

    sc = _subset()
    sc["gates"]["skew_heavy_block_count"] = 2
    with pytest.raises(SidecarSemanticError, match="skew_heavy_block_count"):
        validate_sidecar(sc)

    sc = _subset()
    sc["gates"]["tail_measured_waves"] = 1
    with pytest.raises(SidecarSemanticError, match="tail_measured_waves"):
        validate_sidecar(sc)


def test_subset_budget_and_gate_evidence_must_agree():
    sc = _subset()
    sc["selection_budget"]["certainty_bytes"] = 801
    with pytest.raises(SidecarSemanticError, match="2/5"):
        validate_sidecar(sc)

    sc = _subset()
    sc["gates"]["max_trace_bytes"] = 1999
    with pytest.raises(SidecarSemanticError, match="max_trace_bytes"):
        validate_sidecar(sc)

    sc = _subset()
    sc["bytes"]["raw"] = 2001
    with pytest.raises(SidecarSemanticError, match="raw payload"):
        validate_sidecar(sc)


def test_subset_auxiliary_cannot_select_more_than_full_population():
    sc = _subset()
    sc["selection_auxiliary"]["selected"] = 401
    with pytest.raises(SidecarSemanticError, match="selected exceeds"):
        validate_sidecar(sc)


def test_subset_selected_block_hash_is_membership_bound():
    sc = _subset()
    sc["hashes"]["selected_block_ids_sha256"] = "0" * 64
    with pytest.raises(SidecarSemanticError, match="selected_block_ids_sha256"):
        validate_sidecar(sc)


def test_subset_segments_must_be_complete_reconstructed_cohorts():
    sc = _subset()
    sc["segments"][3]["block_ids"] = [6, 7, 9]
    sc["hashes"]["selected_block_ids_sha256"] = hashlib.sha256(
        b"[0,1,2,3,4,5,6,7,9,57,58,59]").hexdigest()
    with pytest.raises(SidecarSemanticError, match="complete reconstructed"):
        validate_sidecar(sc)


def test_subset_requires_an_exhaustive_grid_roster_and_exact_residency():
    sc = _subset()
    sc["per_sm_slot_map"][0]["block_ids"].pop()
    with pytest.raises(SidecarSemanticError, match="exhaust"):
        validate_sidecar(sc)

    sc = _subset()
    sc["wave_model"]["per_sm_resident_blocks"][0]["resident_blocks"] = 2
    with pytest.raises(SidecarSemanticError, match="n_waves"):
        validate_sidecar(sc)


def test_reduced_shape_requires_variant_and_matched_label():
    s = _stats()
    sc = build_sidecar(
        cell_id="c", kernel_name="_Z1kv", args_hash="ab", stats=s, gate=_Gate(),
        compression=_Comp(), records_captured=400, tracer_version=5,
        stats_pass_version=1, method="reduced_shape_matched_label",
        reason="gate_forced", reduced_shape_variant="grid_stride_footprint_preserving")
    validate_sidecar(sc)
    assert sc["label_pairing"] == "matched_reduced_shape_label"
    # Without the variant the reduced-shape sidecar is rejected.
    del sc["shape"]["reduced_shape_variant"]
    assert not is_valid(sc)


def test_unavailable_forbids_estimator():
    s = _stats()
    sc = build_sidecar(
        cell_id="c", kernel_name="_Z1kv", args_hash="ab", stats=s, gate=_Gate(),
        compression=_Comp(), records_captured=0, tracer_version=5,
        stats_pass_version=1, method="unavailable", reason="gate_forced")
    validate_sidecar(sc)  # no estimator emitted
    sc["estimator"] = {"mode": "weighted_full_shape_estimate", "targets": [],
                       "not_recoverable": ["cycles"],
                       "convergence": {"status": "unavailable"}}
    assert not is_valid(sc)  # estimator on an unavailable cell is rejected
