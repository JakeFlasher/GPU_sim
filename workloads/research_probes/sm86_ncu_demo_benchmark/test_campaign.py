from __future__ import annotations

import csv
from pathlib import Path

import pytest

from workloads.research_probes.sm86_ncu_demo_benchmark import campaign


def _spec() -> dict:
    return campaign._load_json(campaign.DEFAULT_SPEC)


def _workload() -> dict:
    return _spec()["workloads"][0]


def _wide_csv(path: Path, *, duplicate: bool = False) -> None:
    metrics = [
        "gpc__cycles_elapsed.max",
        "gpu__time_duration.sum",
        "l1tex__t_sectors_lookup_hit.sum",
        "l1tex__t_sectors_lookup_miss.sum",
        "dram__sectors_read.sum",
    ]
    header = [
        "ID",
        "Process Name",
        "Kernel Name",
        "Grid Size",
        "Block Size",
        "Device",
        "CC",
        *metrics,
    ]
    units = ["", "", "", "", "", "", "", "cycle", "us", "sector", "sector", "sector"]
    row = [
        "0",
        "gemm_hmma_bf16_sm80_run",
        "gemm_hmma_bf16_sm80_kernel",
        "(16, 32, 1)",
        "(32, 1, 1)",
        "0",
        "8.6",
        "100",
        "1.5",
        "25",
        "75",
        "64",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerow(units)
        writer.writerow(row)
        if duplicate:
            writer.writerow(row)


def test_plan_binds_ncu_cache_policy_and_metric_projection(tmp_path: Path) -> None:
    spec = _spec()
    workload = _workload()
    ncu = campaign._stage_command(spec, workload, "ncu", tmp_path)
    assert ncu[ncu.index("--cache-control") + 1] == "all"
    assert ncu[ncu.index("--replay-mode") + 1] == "application"
    assert ncu[ncu.index("--launch-skip") + 1] == "1"
    assert ncu[-5:] == ["--timing_shape", "--warmup", "1", "--reps", "1"]
    collected = ncu[ncu.index("--metrics") + 1]
    assert "l1tex__t_sectors_lookup_hit.sum" in collected
    assert "lts__t_sectors_lookup_miss.sum" in collected
    assert "lts__t_sector_hit_rate.pct" not in collected
    simulator = campaign._stage_command(spec, workload, "simulator", tmp_path)
    assert simulator[-1] == "--ncu-shaped-metrics"


def test_ncu_parser_preserves_launch_identity_and_normalizes_duration(tmp_path: Path) -> None:
    path = tmp_path / "ncu.csv"
    _wide_csv(path)
    requested = [
        "gpc__cycles_elapsed.max",
        "gpu__time_duration.sum",
        "l1tex__t_sectors_lookup_hit.sum",
        "l1tex__t_sectors_lookup_miss.sum",
        "dram__sectors_read.sum",
    ]
    metrics, identity = campaign._parse_ncu_metrics(path, requested)
    assert metrics["gpu__time_duration.sum"]["value"] == 1500.0
    assert metrics["gpu__time_duration.sum"]["unit"] == "ns"
    derived = campaign._derive_ncu_metrics(
        metrics,
        {
            "metrics": [
                {"id": "cycles", "ncu_metric": "gpc__cycles_elapsed.max"},
                {
                    "id": "l1",
                    "ncu_metric": "l1tex__t_sector_hit_rate.pct",
                    "ncu_sources": [
                        "l1tex__t_sectors_lookup_hit.sum",
                        "l1tex__t_sectors_lookup_miss.sum",
                    ],
                    "ncu_formula": "100 * hit_sectors / (hit_sectors + miss_sectors)",
                },
            ]
        },
    )
    assert derived["l1tex__t_sector_hit_rate.pct"]["value"] == 25.0
    assert derived["l1tex__t_sector_hit_rate.pct"]["status"] == "measured"
    metrics["l1tex__t_sectors_lookup_hit.sum"]["value"] = 0.0
    metrics["l1tex__t_sectors_lookup_miss.sum"]["value"] = 0.0
    unavailable = campaign._derive_ncu_metrics(
        metrics,
        {
            "metrics": [
                {
                    "id": "l1",
                    "ncu_metric": "l1tex__t_sector_hit_rate.pct",
                    "ncu_sources": [
                        "l1tex__t_sectors_lookup_hit.sum",
                        "l1tex__t_sectors_lookup_miss.sum",
                    ],
                    "ncu_formula": "100 * hit_sectors / (hit_sectors + miss_sectors)",
                }
            ]
        },
    )
    assert unavailable["l1tex__t_sector_hit_rate.pct"]["value"] is None
    assert (
        unavailable["l1tex__t_sector_hit_rate.pct"]["status"]
        == "unavailable_zero_denominator"
    )
    assert identity["Kernel Name"] == "gemm_hmma_bf16_sm80_kernel"
    assert identity["Grid Size"] == "(16, 32, 1)"


def test_ncu_parser_rejects_duplicate_launch_rows(tmp_path: Path) -> None:
    path = tmp_path / "ncu.csv"
    _wide_csv(path, duplicate=True)
    with pytest.raises(campaign.CampaignError, match="has 2 rows"):
        campaign._parse_ncu_metrics(path, ["gpc__cycles_elapsed.max"])


def test_gpu_and_simulator_environments_remove_ambient_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/tmp/ambient.so")
    monkeypatch.setenv("SIM_RUN_PERFORMANCE_COUNTERS", "1")
    gpu = campaign._clean_environment()
    simulator = campaign._clean_environment(simulator=True)
    assert "CUDA_VISIBLE_DEVICES" not in gpu
    assert "CUDA_INJECTION64_PATH" not in gpu
    assert gpu["LD_LIBRARY_PATH"] == "/opt/cuda/lib64"
    assert "SIM_RUN_PERFORMANCE_COUNTERS" not in simulator


def test_kernel_join_only_removes_tracer_function_suffix() -> None:
    assert (
        campaign._normalized_kernel_name("gemm_hmma_bf16_sm80_kernel___0")
        == "gemm_hmma_bf16_sm80_kernel"
    )
    assert campaign._normalized_kernel_name("kernel__part") == "kernel__part"


def test_repository_artifacts_use_portable_paths() -> None:
    path = campaign.REPO_ROOT / "arch_profiles/cards/sm_86.json"
    assert campaign._portable_path(path) == "arch_profiles/cards/sm_86.json"
