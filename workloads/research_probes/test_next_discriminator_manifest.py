from __future__ import annotations

import json
from pathlib import Path

from workloads.research_probes.ldgsts_burst_release.campaign import configured_cells


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/sm86_ldgsts_burst_release_next_discriminator.json"


def test_next_discriminator_is_reproducible_and_frozen() -> None:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["status"] == "prospective_not_run"
    matrix = document["matrix"]
    assert matrix["cell_count"] == 40
    assert matrix["cell_selector"].split(",") == [
        cell.key for cell in configured_cells()
    ]
    assert matrix["launch_geometry"] == {
        "cta_count": 1,
        "threads_per_cta": 128,
        "shared_bytes": 32768,
    }
    commands = document["commands"]
    assert commands["build"]["argv"] == [
        "make",
        "-C",
        "workloads/research_probes/ldgsts_burst_release",
        "clean",
        "all",
    ]
    assert commands["nvbit"]["argv"] == [
        "workloads/research_probes/ldgsts_burst_release/ldgsts_burst_release",
        "--mode",
        "direct",
        "--assay",
        "ldgsts",
        "--cells",
        "ldgsts_w1_l1_d4_g0",
        "--bursts",
        "16384",
        "--warmup",
        "64",
        "--repetition",
        "1",
        "--order",
        "forward",
    ]
    assert commands["simulator"]["argv"][-2:] == ["--mode", "l1-l2"]
    assert commands["ncu"]["comparison_only"] is True
    assert document["source"]["pairing"]["independent_retirement_observable"]
    assert document["decision_rule"]["pass_or_identify"]
    assert document["decision_rule"]["falsify_or_no_go"]
    assert document["decision_rule"]["unknown_gate"]
    assert document["evidence_boundary"]["ncu_values_used_as_simulator_inputs"] is False
