from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.local_labels import admit_cutlass_successor_capture as admit
from tools.local_labels import build_cutlass_successor_cohort as cohort
from tools.local_labels import parse_ncu_metrics


ROOT = Path(__file__).resolve().parents[2]
RECEIPT_SCHEMA = ROOT / admit.RECEIPT_SCHEMA_REL
OVERLAY_SCHEMA = ROOT / admit.OVERLAY_SCHEMA_REL
MATRIX = json.loads((ROOT / cohort.MATRIX_REL).read_text())
ROSTER = json.loads((ROOT / cohort.ROSTER_REL).read_text())
ROSTER_ROWS = {
    (row["matrix_key"], row["variant_id"]): row for row in ROSTER["rows"]
}
GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"
OTHER_GPU_UUID = "GPU-22222222-2222-2222-2222-222222222222"


_REPORT_CSV = '''ID,Kernel Name,Block Size,Grid Size,Device,Metric Name,Metric Value
0,_Zsynthetic_kernel,"32,1,1","1,1,1",NVIDIA GeForce RTX 3090 Ti,gpc__cycles_elapsed.max,1
0,_Zsynthetic_kernel,"32,1,1","1,1,1",NVIDIA GeForce RTX 3090 Ti,gpu__time_duration.sum,1
0,_Zsynthetic_kernel,"32,1,1","1,1,1",NVIDIA GeForce RTX 3090 Ti,lts__t_sector_hit_rate.pct,50
0,_Zsynthetic_kernel,"32,1,1","1,1,1",NVIDIA GeForce RTX 3090 Ti,l1tex__t_sector_hit_rate.pct,50
'''
_REPORT_IDENTITY_COLUMNS = [
    "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
]
_REPORT_IDENTITY = {
    "ID": "0",
    "Device": "NVIDIA GeForce RTX 3090 Ti",
    "Kernel Name": "_Zsynthetic_kernel",
    "Block Size": "32,1,1",
    "Grid Size": "1,1,1",
}


def _row_report_identity(row: dict) -> dict[str, str]:
    launch = row["shape"]["trace_shape"]["launch_config"]
    return {
        "ID": "0",
        "Device": admit.GPU_NAME,
        "Kernel Name": _REPORT_IDENTITY["Kernel Name"],
        "Block Size": ",".join(str(value) for value in launch["block"]),
        "Grid Size": ",".join(str(value) for value in launch["grid"]),
    }


def _row_report_csv(row: dict) -> str:
    identity = _row_report_identity(row)
    prefix = (
        f'{identity["ID"]},{identity["Kernel Name"]},'
        f'"{identity["Block Size"]}","{identity["Grid Size"]}",'
        f'{identity["Device"]}'
    )
    return (
        "ID,Kernel Name,Block Size,Grid Size,Device,Metric Name,Metric Value\n"
        f"{prefix},gpc__cycles_elapsed.max,1\n"
        f"{prefix},gpu__time_duration.sum,1\n"
        f"{prefix},lts__t_sector_hit_rate.pct,50\n"
        f"{prefix},l1tex__t_sector_hit_rate.pct,50\n"
    )


def _row_for_report_path(path: Path) -> dict:
    match = re.fullmatch(r"(\d{2})_ncu_0\.bin", path.name)
    if match is not None:
        return MATRIX["rows"][int(match.group(1))]
    if path.name == "run_0_launch_set.ncu-rep":
        slug = path.parent.parent.name
        return next(
            row for row in MATRIX["rows"] if row["successor_slug"] == slug
        )
    raise AssertionError(f"unexpected synthetic NCU report path: {path}")


_REAL_IMPORT_NCU_REPORT_CSV = admit._import_ncu_report_csv


@pytest.fixture(autouse=True)
def _mock_ncu_report_import(monkeypatch):
    def imported(_identity, path):
        return (
            _row_report_csv(_row_for_report_path(path)),
            admit.sha256_bytes(path.read_bytes()),
            None,
        )

    monkeypatch.setattr(admit, "_import_ncu_report_csv", imported)
    monkeypatch.setattr(
        admit,
        "_sass_kernel_names",
        lambda _row: {_REPORT_IDENTITY["Kernel Name"]},
    )


def test_bundle_validator_rejects_alternate_second_generation(monkeypatch):
    documents = list(cohort.validate_existing_bundle())
    alternate = copy.deepcopy(documents)
    alternate[3]["rows"][0]["successor_slug"] = "alternate_valid_generation"
    monkeypatch.setattr(
        admit.cohort,
        "validate_existing_bundle",
        lambda: tuple(alternate),
    )

    with pytest.raises(
        admit.AdmissionError,
        match="differ from pinned Git bytes",
    ):
        admit.validate_immutable_bundle()


def test_successor_reimport_uses_descriptor_bound_ncu_authority(
    tmp_path: Path, monkeypatch,
):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    identity = admit.ncu_executor.ExpectedNcuIdentity(
        "/opt/nvidia/ncu",
        "b" * 64,
        "Version 2026.1.0.0 (build synthetic)",
    )
    observed = {}

    def imported(expected, path, *, require_cgroup_containment):
        observed.update({
            "expected": expected,
            "path": path,
            "require_cgroup_containment": require_cgroup_containment,
        })
        return (
            parse_ncu_metrics.ImportedReportCsv(
                text=_REPORT_CSV,
                report_sha256=admit.sha256_bytes(report.read_bytes()),
                report_size_bytes=len(report.read_bytes()),
            ),
            None,
        )

    monkeypatch.setattr(parse_ncu_metrics, "_import_report_csv", imported)
    csv_text, report_sha256, error = _REAL_IMPORT_NCU_REPORT_CSV(
        identity, report
    )
    assert error is None
    assert csv_text == _REPORT_CSV
    assert report_sha256 == admit.sha256_bytes(report.read_bytes())
    assert observed == {
        "expected": identity,
        "path": report,
        "require_cgroup_containment": True,
    }



def _report_sidecar() -> dict:
    return {
        "cache_hit_rate_l2_sector": 0.5,
        "cache_hit_rate_l1tex_sector": 0.5,
        "confidence_interval_l2_sector": {
            "lower": 0.5,
            "upper": 0.5,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
        "confidence_interval_l1tex_sector": {
            "lower": 0.5,
            "upper": 0.5,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
        "unit_tagged_metrics": [
            {"ncu_metric": "gpc__cycles_elapsed.max", "unit": "cycles", "value": 1.0},
            {"ncu_metric": "gpu__time_duration.sum", "unit": "ns", "value": 1.0},
        ]
    }


def _report_observation() -> dict:
    observation = {
        "identity_columns": list(_REPORT_IDENTITY_COLUMNS),
        "identity": dict(_REPORT_IDENTITY),
        "report_sha256": "unused",
    }
    return observation


def _report_launch_authority() -> tuple[dict[str, object], set[str]]:
    expected: dict[str, object] = {
        "ID": _REPORT_IDENTITY["ID"],
        "Device": _REPORT_IDENTITY["Device"],
        "Block Size": [32, 1, 1],
        "Grid Size": [1, 1, 1],
    }
    return expected, {_REPORT_IDENTITY["Kernel Name"]}


def test_ncu_report_import_failure_is_not_sidecar_only(tmp_path: Path, monkeypatch):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (
            None,
            None,
            type("E", (), {"detail": "ncu unavailable"})(),
        ),
    )
    with pytest.raises(admit.AdmissionError, match="import failed"):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            admit.sha256_bytes(report.read_bytes()),
            _report_sidecar(),
            _report_observation(),
            *_report_launch_authority(),
        )


def test_ncu_report_import_identity_drift_is_rejected(
    tmp_path: Path, monkeypatch,
):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (_REPORT_CSV, "0" * 64, None),
    )
    with pytest.raises(admit.AdmissionError, match="different retained bytes"):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            admit.sha256_bytes(report.read_bytes()),
            _report_sidecar(),
            _report_observation(),
            *_report_launch_authority(),
        )


def test_ncu_report_metric_disagreement_is_rejected(tmp_path: Path, monkeypatch):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    csv = "Kernel Name,Metric Name,Metric Value\n_Zsynthetic_kernel,gpc__cycles_elapsed.max,2\n_Zsynthetic_kernel,gpu__time_duration.sum,1\n"
    report_sha256 = admit.sha256_bytes(report.read_bytes())
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (csv, report_sha256, None),
    )
    with pytest.raises(admit.AdmissionError, match="metric disagreement"):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            report_sha256,
            _report_sidecar(),
            _report_observation(),
            *_report_launch_authority(),
        )


@pytest.mark.parametrize(
    ("sidecar_field", "metric"),
    [
        ("cache_hit_rate_l2_sector", "lts__t_sector_hit_rate.pct"),
        ("cache_hit_rate_l1tex_sector", "l1tex__t_sector_hit_rate.pct"),
    ],
)
def test_ncu_report_percentage_disagreement_is_rejected(
    tmp_path: Path,
    monkeypatch,
    sidecar_field: str,
    metric: str,
):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    report_sha256 = admit.sha256_bytes(report.read_bytes())
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (_REPORT_CSV, report_sha256, None),
    )
    sidecar = _report_sidecar()
    sidecar[sidecar_field] = 0.25

    with pytest.raises(
        admit.AdmissionError,
        match=rf"percentage disagreement: {re.escape(metric)}",
    ):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            report_sha256,
            sidecar,
            _report_observation(),
            *_report_launch_authority(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_metric",
        "confidence_interval",
        "unexpected_diagnostics",
    ],
)
def test_ncu_report_closes_all_sidecar_derived_fields(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    report_sha256 = admit.sha256_bytes(report.read_bytes())
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (_REPORT_CSV, report_sha256, None),
    )
    sidecar = _report_sidecar()
    if mutation == "extra_metric":
        sidecar["unit_tagged_metrics"].append({
            "ncu_metric": "sm__cycles_elapsed.avg",
            "unit": "cycles",
            "value": 1.0,
        })
        expected = "metric disagreement"
    elif mutation == "confidence_interval":
        sidecar["confidence_interval_l2_sector"]["lower"] = 0.25
        expected = "percentage disagreement"
    else:
        sidecar["parser_diagnostics"] = {
            "ncu_sector_hit_rate_clamped_to_100": [
                {
                    "metric_label": "lts__t_sector_hit_rate.pct",
                    "raw_value": 101.0,
                }
            ]
        }
        expected = "unreported parser diagnostics"

    with pytest.raises(admit.AdmissionError, match=expected):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            report_sha256,
            sidecar,
            _report_observation(),
            *_report_launch_authority(),
        )


def test_ncu_report_requires_all_parser_diagnostics(
    tmp_path: Path,
    monkeypatch,
):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    report_sha256 = admit.sha256_bytes(report.read_bytes())
    csv_text = _REPORT_CSV.replace(
        "lts__t_sector_hit_rate.pct,50",
        "lts__t_sector_hit_rate.pct,101",
    )
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (csv_text, report_sha256, None),
    )
    sidecar = _report_sidecar()
    sidecar["cache_hit_rate_l2_sector"] = 1.0
    sidecar["confidence_interval_l2_sector"]["lower"] = 1.0
    sidecar["confidence_interval_l2_sector"]["upper"] = 1.0

    with pytest.raises(admit.AdmissionError, match="parser diagnostics"):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            report_sha256,
            sidecar,
            _report_observation(),
            *_report_launch_authority(),
        )


def test_ncu_report_multiple_launches_are_rejected(tmp_path: Path, monkeypatch):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    csv = '''ID,Kernel Name,Block Size,Grid Size,Device,Metric Name,Metric Value
0,_Zsynthetic_kernel,"32,1,1","1,1,1",NVIDIA GeForce RTX 3090 Ti,gpc__cycles_elapsed.max,1
1,_Zsynthetic_kernel,"32,1,1","1,1,1",NVIDIA GeForce RTX 3090 Ti,gpc__cycles_elapsed.max,1
0,_Zsynthetic_kernel,"32,1,1","1,1,1",NVIDIA GeForce RTX 3090 Ti,gpu__time_duration.sum,1
1,_Zsynthetic_kernel,"32,1,1","1,1,1",NVIDIA GeForce RTX 3090 Ti,gpu__time_duration.sum,1
'''
    report_sha256 = admit.sha256_bytes(report.read_bytes())
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (csv, report_sha256, None),
    )
    with pytest.raises(admit.AdmissionError, match="metric extraction failed"):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            report_sha256,
            _report_sidecar(),
            _report_observation(),
            *_report_launch_authority(),
        )


def test_ncu_report_launch_identity_must_match_successor_authority(
    tmp_path: Path, monkeypatch,
):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    identity = dict(_REPORT_IDENTITY)
    identity["Grid Size"] = "2,1,1"
    prefix = (
        f'{identity["ID"]},{identity["Kernel Name"]},'
        f'"{identity["Block Size"]}","{identity["Grid Size"]}",'
        f'{identity["Device"]}'
    )
    csv_text = (
        "ID,Kernel Name,Block Size,Grid Size,Device,Metric Name,Metric Value\n"
        f"{prefix},gpc__cycles_elapsed.max,1\n"
        f"{prefix},gpu__time_duration.sum,1\n"
        f"{prefix},lts__t_sector_hit_rate.pct,50\n"
        f"{prefix},l1tex__t_sector_hit_rate.pct,50\n"
    )
    report_sha256 = admit.sha256_bytes(report.read_bytes())
    observation = _report_observation()
    observation["identity"] = identity
    observation["report_sha256"] = report_sha256
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (csv_text, report_sha256, None),
    )
    with pytest.raises(admit.AdmissionError, match="successor launch"):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            report_sha256,
            _report_sidecar(),
            observation,
            *_report_launch_authority(),
        )


def test_ncu_report_kernel_must_exist_in_pinned_sass_authority(
    tmp_path: Path, monkeypatch,
):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    report_sha256 = admit.sha256_bytes(report.read_bytes())
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (_REPORT_CSV, report_sha256, None),
    )
    observation = _report_observation()
    observation["report_sha256"] = report_sha256
    expected_launch, _allowed = _report_launch_authority()
    with pytest.raises(admit.AdmissionError, match="pinned SASS"):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            report_sha256,
            _report_sidecar(),
            observation,
            expected_launch,
            {"_Zother_kernel"},
        )


def test_ncu_report_normalizes_supported_device_and_dimension_forms(
    tmp_path: Path, monkeypatch,
):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    identity = {
        **_REPORT_IDENTITY,
        "Device": "0",
        "Block Size": "(32, 1, 1)",
        "Grid Size": "(1, 1, 1)",
    }
    prefix = (
        f'{identity["ID"]},{identity["Kernel Name"]},'
        f'"{identity["Block Size"]}","{identity["Grid Size"]}",'
        f'{identity["Device"]}'
    )
    csv_text = (
        "ID,Kernel Name,Block Size,Grid Size,Device,Metric Name,Metric Value\n"
        f"{prefix},gpc__cycles_elapsed.max,1\n"
        f"{prefix},gpu__time_duration.sum,1\n"
        f"{prefix},lts__t_sector_hit_rate.pct,50\n"
        f"{prefix},l1tex__t_sector_hit_rate.pct,50\n"
    )
    report_sha256 = admit.sha256_bytes(report.read_bytes())
    observation = _report_observation()
    observation["identity"] = identity
    observation["report_sha256"] = report_sha256
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (csv_text, report_sha256, None),
    )

    observed = admit._verify_ncu_report(
        _expected_ncu_identity(),
        report,
        report_sha256,
        _report_sidecar(),
        observation,
        *_report_launch_authority(),
    )

    assert observed == {
        "kernel_name": _REPORT_IDENTITY["Kernel Name"],
        "grid_dim": [1, 1, 1],
        "block_dim": [32, 1, 1],
    }


def test_nvbit_kernel_variant_join_is_explicit_and_fail_closed():
    assert admit._nvbit_kernel_base("_Zsynthetic_kernel___0") == (
        "_Zsynthetic_kernel"
    )
    for value in ("_Zsynthetic_kernel", "_Zsynthetic_kernel___1"):
        with pytest.raises(admit.AdmissionError):
            admit._nvbit_kernel_base(value)


def _seal(document: dict) -> dict:
    document = copy.deepcopy(document)
    document["receipt_seal_sha256"] = admit.canonical_sha(
        {key: value for key, value in document.items() if key != "receipt_seal_sha256"}
    )
    return document


def _process(pid: int, command: list[str]) -> dict:
    smi_stdout = "NVIDIA GeForce RTX 3090 Ti, 8.6, Enabled, 1350, 10501\n"
    identity_stdout = f"{GPU_UUID}, 610.43.03\n"
    process = {
        "host_identity_sha256": admit.canonical_sha({
            "gpu": "NVIDIA GeForce RTX 3090 Ti",
            "smi": admit.sha256_bytes(smi_stdout.encode()),
            "gpu_uuid_sha256": admit.sha256_bytes(GPU_UUID.encode()),
            "driver_version": "610.43.03",
            "identity_stdout_sha256": admit.sha256_bytes(
                identity_stdout.encode()
            ),
        }),
        "boot_id_sha256": "b" * 64,
        "pid": pid,
        "started_at_utc": f"2026-07-23T00:0{pid % 10}:00Z",
        "command_sha256": admit.canonical_sha(command),
        "containment": {
            "kind": admit.delegated_cgroup.CGROUP_V2_KILL_V1,
            "parent_cgroup_sha256": "d" * 64,
            "cgroup_inode": 1000 + pid,
            "terminal_populated_zero": True,
            "leaf_removed": True,
        },
    }
    process["process_identity_sha256"] = admit.canonical_sha(process)
    return process


def _ncu_authority() -> dict:
    return {
        "status": "ok",
        "expected": "2026.1.0",
        "execution_mode": "direct_inner_elf_v1",
        "host_machine": "x86_64",
        "launcher_resolved_path": admit.NCU_LAUNCHER_PATH,
        "launcher_sha256": admit.NCU_LAUNCHER_SHA256,
        "launcher_target_relpath": (
            admit.campaign_preflight.NCU_LAUNCHER_TARGET_RELPATH
        ),
        "resolved_path": admit.NCU_EXECUTABLE_PATH,
        "executable_sha256": admit.NCU_EXECUTABLE_SHA256,
        "version_line": admit.NCU_VERSION_LINE,
    }


def _expected_ncu_identity() -> admit.ncu_executor.ExpectedNcuIdentity:
    return admit._validate_ncu_authority(_ncu_authority())


def _environment_payload(
    root: Path, row: dict, kind: str, preflight: dict, index: int
) -> bytes:
    live_outputs = {
        "smi_stdout": "NVIDIA GeForce RTX 3090 Ti, 8.6, Enabled, 1350, 10501\n",
        "competing_process_stdout": "",
        "identity_stdout": f"{GPU_UUID}, 610.43.03\n",
        "nvcc_stdout": "Cuda compilation tools, release 13.2, V13.2.0\n",
    }
    live = {
        "smi_command": ["nvidia-smi", "--query-gpu=name,compute_cap,persistence_mode,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"],
        "smi_stdout_sha256": admit.sha256_bytes(live_outputs["smi_stdout"].encode()),
        "competing_process_command": ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        "competing_process_stdout_sha256": admit.sha256_bytes(live_outputs["competing_process_stdout"].encode()),
        "identity_command": ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
        "identity_stdout_sha256": admit.sha256_bytes(live_outputs["identity_stdout"].encode()),
        "gpu_uuid_sha256": admit.sha256_bytes(GPU_UUID.encode()),
        "driver_version": "610.43.03",
        "nvcc_command": ["/usr/local/cuda-13.2/bin/nvcc", "--version"],
        "nvcc_stdout_sha256": admit.sha256_bytes(live_outputs["nvcc_stdout"].encode()),
        "cuda_release": {"major": 13, "minor": 2},
        "nvbit_version": "1.8",
        "gpu_name": "NVIDIA GeForce RTX 3090 Ti",
        "persistence_mode": "Enabled",
        "sm_clock_mhz": 1350,
        "mem_clock_mhz": 10501,
    }
    if kind == "ncu":
        ncu_authority = _ncu_authority()
        live_outputs["ncu_stdout"] = (
            "NVIDIA (R) Nsight Compute Command Line Profiler\n"
            "Copyright (c) 2018-2026 NVIDIA Corporation\n"
            + ncu_authority["version_line"]
            + "\n"
        )
        live.update({
            "ncu_command": [ncu_authority["resolved_path"], "--version"],
            "ncu_stdout_sha256": admit.sha256_bytes(
                live_outputs["ncu_stdout"].encode()
            ),
            "ncu_authority": ncu_authority,
            "ncu_version": "2026.1.0",
        })
        selected = admit.ncu_executor.NcuCollectionEnvironment({
            "LANG": "C",
            "LANGUAGE": "C",
            "LC_ALL": "C",
            "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
            "PWD": str(
                (ROOT / row["static_launch_inventory"]["binary_path"])
                .parent.resolve(strict=True)
            ),
            "PYTHONNOUSERSITE": "1",
            "LABEL_DRIVER_GPU_INDEX": "0",
            "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
            "LD_SOURCE_COMMIT_OVERRIDE": admit.PINNED_BUNDLE_COMMIT,
        }).as_dict()
    else:
        selected = admit._expected_nvbit_environment(root, row)
    return json.dumps({
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_environment",
        "source_commit": admit.PINNED_BUNDLE_COMMIT,
        "preflight_pins_sha256": admit.canonical_sha(preflight),
        "live_checks": live,
        "live_outputs": live_outputs,
        "environment": selected,
    }, sort_keys=True).encode()


def _write_row_trace(root: Path, row: dict, *, kernel_name: str) -> None:
    from tools.nvbit_tracer.proto_binding_loader import (
        load_dynamic_trace_bindings,
    )
    from tools.nvbit_tracer.test_trace_admission import _write_trace

    _write_trace(root)
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    trace = trace_pb2.Trace()
    path = root / "dynamic_trace.pb"
    trace.ParseFromString(path.read_bytes())
    trace.binary_version = admit.NVBIT_TRACE_BINARY_VERSION
    trace.nvbit_version = admit.NVBIT_VERSION
    kernel = trace.gpu_device[0].streams[0].kernels[0]
    kernel.name = kernel_name
    launch = row["shape"]["trace_shape"]["launch_config"]
    kernel.grid_dim.x, kernel.grid_dim.y, kernel.grid_dim.z = launch["grid"]
    kernel.block_dim.x, kernel.block_dim.y, kernel.block_dim.z = launch["block"]
    path.write_bytes(trace.SerializeToString())


def _receipt(root: Path, row: dict, kind: str, index: int) -> Path:
    slug = row["successor_slug"]
    identity = {
        "slug": slug,
        "shape_point": row["shape"]["shape_point"],
        "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
        "source_commit": admit.PINNED_BUNDLE_COMMIT,
        "kernel_signature_sha256": hashlib.sha256(
            f"{slug}|{row['static_launch_inventory']['binary_path']}".encode()
        ).hexdigest(),
        "arch": row["arch"],
        "ncu_metric_version": admit.DEFAULT_METRIC_VERSION,
    }
    launch = {
        "launch_index": 0,
        "kernel_name": slug,
        "kernel_signature_sha256": identity["kernel_signature_sha256"],
        "grid_dim": row["shape"]["trace_shape"]["launch_config"]["grid"],
        "block_dim": row["shape"]["trace_shape"]["launch_config"]["block"],
    }
    launch["launch_identity_sha256"] = admit.canonical_sha(launch)
    if kind == "ncu":
        ncu_authority = _ncu_authority()
        ncu_identity = admit._validate_ncu_authority(ncu_authority)
        command = admit._expected_ncu_command(root, row, ncu_identity)
    else:
        command = admit._expected_nvbit_command(row)
    process = _process(100 + index * 2 + (kind == "nvbit"), command)
    roles = (
        ["ncu_report", "label_sidecar", "label_run_manifest", "ncu_launch_observation", "preflight", "environment"]
        if kind == "ncu"
        else ["dynamic_trace", "stats", "threadblock", "trace_summary", "preflight", "environment"]
    )
    preflight_document = admit._expected_preflight(
        row, ROSTER_ROWS[(row["matrix_key"], row["variant_id"])]
    )
    if kind == "nvbit":
        trace_root = root / "nvbit" / slug
        trace_root.parent.mkdir(parents=True, exist_ok=True)
        _write_row_trace(
            trace_root,
            row,
            kernel_name=f'{_REPORT_IDENTITY["Kernel Name"]}___0',
        )
    artifacts = []
    preflight_path = None
    preflight_sha = None
    environment_sha = None
    ncu_artifact_names = {
        "ncu_report": "run_0_launch_set.ncu-rep",
        "label_sidecar": "label_0_0.json",
        "label_run_manifest": "label_run_manifest_0_launch_set.json",
        "ncu_launch_observation": "run_0_launch_set.launch_observation.json",
    }
    for role_index, role in enumerate(roles):
        relative = (
            f"{row['arch']}/{slug}/trace_00/{ncu_artifact_names[role]}"
            if kind == "ncu" and role in ncu_artifact_names
            else (
                f"artifacts/{index:02d}_{kind}_{role_index}.bin"
                if kind == "ncu"
                else (
                    f"nvbit/{slug}/threadblocks/device_0/stream_0/kernel_0/0.pb"
                    if role == "threadblock"
                    else (
                        f"nvbit/{slug}/{'dynamic_trace.pb' if role == 'dynamic_trace' else 'stats.csv' if role == 'stats' else 'trace_summary.json' if role == 'trace_summary' else f'{role_index}_{role}.bin'}"
                    )
                )
            )
        )
        payload = f"synthetic-{index}-{kind}-{role}".encode()
        if role == "preflight":
            payload = json.dumps(preflight_document, sort_keys=True).encode()
        elif role == "environment":
            payload = _environment_payload(root, row, kind, preflight_document, index)
        elif kind == "nvbit" and role in {"dynamic_trace", "stats", "threadblock"}:
            payload = (root / relative).read_bytes()
        elif kind == "ncu" and role == "label_sidecar":
            payload = json.dumps({
                "schema_version": 5,
                "device": row["arch"],
                "slug": slug,
                "shape_point": row["shape"]["shape_point"],
                "repeat_index": 0,
                "launch_index": 0,
                "binary_sha256": identity["binary_sha256"],
                "profiler_family": "nsight_compute",
                "profiler_version": admit.NCU_VERSION_LINE,
                "cuda_version": "13.2",
                "driver_version": "610.43.03",
                "gpu_uuid_sha256": admit.sha256_bytes(GPU_UUID.encode()),
                "replay_mode": "kernel",
                "aggregation": "kernel",
                "warmup_count": 0,
                "evidence_class": "campaign",
                "cache_hit_rate_l2_sector": 0.5,
                "cache_hit_rate_l1tex_sector": 0.5,
                "confidence_interval_l2_sector": {
                    "lower": 0.5, "upper": 0.5, "confidence_level": 0.95,
                    "units": "sector_hit_rate_fraction", "aggregation": "kernel",
                },
                "confidence_interval_l1tex_sector": {
                    "lower": 0.5, "upper": 0.5, "confidence_level": 0.95,
                    "units": "sector_hit_rate_fraction", "aggregation": "kernel",
                },
                "measurement_protocol": {
                    "ncu_clock_control": "none",
                    "ncu_cache_control": "all",
                    "ncu_replay_mode": "kernel",
                    "ncu_target_processes": "all",
                    "ncu_filter_mode": "global",
                    "ncu_profile_from_start": False,
                    "ncu_launch_skip": 0,
                    "ncu_launch_count": 1,
                    "host_gpu_index": 0,
                    "host_persistence_mode": "Enabled",
                    "host_sm_clock_mhz": 1350,
                    "host_mem_clock_mhz": 10501,
                },
                "unit_tagged_metrics": [
                    {"ncu_metric": "gpc__cycles_elapsed.max", "unit": "cycles", "value": 1},
                    {"ncu_metric": "gpu__time_duration.sum", "unit": "ns", "value": 1},
                ],
            }, sort_keys=True).encode()
        elif kind == "ncu" and role == "label_run_manifest":
            payload = json.dumps({
                "schema_version": 1, "source_commit": admit.PINNED_BUNDLE_COMMIT,
                "binary_sha256": identity["binary_sha256"], "kernel_signature_sha256": identity["kernel_signature_sha256"], "kernel_signature_source": "fallback_legacy", "arch": row["arch"],
                "slug": slug, "shape_point": row["shape"]["shape_point"],
                "repeat_index": 0, "launch_index": 0,
            }, sort_keys=True).encode()
        elif kind == "ncu" and role == "ncu_launch_observation":
            report_payload = f"synthetic-{index}-{kind}-ncu_report".encode()
            observation = {
                "schema_version": 1, "kind": "ncu_details_csv_launch_observation",
                "kernel_name_base": "mangled", "requested_launch_skip": 0,
                "requested_launch_count": 1, "report_sha256": hashlib.sha256(report_payload).hexdigest(),
                "identity_columns": list(_REPORT_IDENTITY_COLUMNS),
                "identity": _row_report_identity(row),
            }
            observation["observation_sha256"] = admit.canonical_sha(observation)
            payload = json.dumps(observation, sort_keys=True).encode()
        elif kind == "nvbit" and role == "trace_summary":
            payload = json.dumps({
                "schema_version": 1,
                "producer": admit.NVBIT_TRACE_SUMMARY_PRODUCER,
                "artifact_kind": "cutlass_successor_nvbit_trace_summary",
                "trace_contract_id": admit.TRACE_CONTRACT_ID,
                "memory_payload_classes": admit.TRACE_MEMORY_PAYLOAD_CLASSES,
                "binary_version": admit.NVBIT_TRACE_BINARY_VERSION,
                "nvbit_version": admit.NVBIT_VERSION,
                "trace_contract_status": "pass", "expected_record_count": 1,
                "admitted_record_count": 1, "threadblock_count": 1,
                "dynamic_trace": "dynamic_trace.pb", "stats": "stats.csv",
            }, sort_keys=True).encode()
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        artifacts.append({"role": role, "path": relative, "sha256": digest, "size_bytes": len(payload)})
        if role == "preflight":
            preflight_path, preflight_sha = relative, digest
        if role == "environment":
            environment_sha = digest
    provenance = {
        "capture_producer_sha256": admit.sha256_bytes(
            (ROOT / "tools/local_labels/capture_cutlass_successor_receipts.py").read_bytes()
        ),
        "containment_supervisor_sha256": admit.sha256_bytes(
            Path(admit.delegated_cgroup.__file__).read_bytes()
        ),
        "phase_tool_sha256": admit.sha256_bytes(
            (ROOT / ("tools/local_labels/label_driver.sh" if kind == "ncu" else "tools/nvbit_tracer/tracer_tool/tracer_tool.so")).read_bytes()
        ),
    }
    if kind == "ncu":
        provenance["action_executor_sha256"] = admit.sha256_bytes(
            Path(admit.ncu_executor.__file__).read_bytes()
        )
    receipt = {
        "schema_version": 2,
        "artifact_kind": "cutlass_successor_capture_receipt",
        "receipt": {
            "capture_kind": kind,
            "status": "pass",
            "matrix_key": row["matrix_key"],
            "variant_id": row["variant_id"],
            "identity": identity,
            "launch_identity": launch,
            "process": process,
            "provenance": provenance,
            "captured_at_utc": "2026-07-23T01:00:00Z",
            "preflight_path": preflight_path,
            "preflight_sha256": preflight_sha,
            "environment_sha256": environment_sha,
            "command": command,
            "command_sha256": admit.canonical_sha(command),
            "semantic_validation": (
                {
                    "semantic_kind": "ncu_metric_validation",
                    "primary_metric": "gpc__cycles_elapsed.max",
                    "secondary_metric": "gpu__time_duration.sum",
                    "primary_metric_present": True,
                    "secondary_metric_present": True,
                    "target_only": True,
                    "simulator_config_present": False,
                }
                if kind == "ncu"
                else {
                    "semantic_kind": "nvbit_trace_validation",
                    "trace_contract_status": "pass",
                    "dynamic_trace_role": "dynamic_trace",
                    "summary_role": "trace_summary",
                    "target_only": True,
                    "simulator_config_present": False,
                }
            ),
            "artifacts": artifacts,
        },
    }
    path = root / f"{index:02d}_{kind}{admit.RECEIPT_SUFFIX}"
    path.write_text(json.dumps(_seal(receipt), indent=2, sort_keys=True) + "\n")
    return path


def _all_pairs(root: Path) -> None:
    for index, row in enumerate(MATRIX["rows"]):
        _receipt(root, row, "ncu", index)
        _receipt(root, row, "nvbit", index)


def _rewrite(path: Path, document: dict) -> None:
    path.write_text(json.dumps(_seal(document), indent=2, sort_keys=True) + "\n")


def _mutate_json_artifact(root: Path, receipt_path: Path, role: str, mutate) -> None:
    document = json.loads(receipt_path.read_text())
    artifact = next(item for item in document["receipt"]["artifacts"] if item["role"] == role)
    path = root / artifact["path"]
    payload = json.loads(path.read_text())
    mutate(payload)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(encoded)
    artifact["sha256"] = admit.sha256_bytes(encoded)
    artifact["size_bytes"] = len(encoded)
    if role == "preflight":
        document["receipt"]["preflight_sha256"] = artifact["sha256"]
    elif role == "environment":
        document["receipt"]["environment_sha256"] = artifact["sha256"]
    _rewrite(receipt_path, document)


def _mutate_dynamic_trace_artifact(
    root: Path, receipt_path: Path, mutate,
) -> None:
    from tools.nvbit_tracer.proto_binding_loader import (
        load_dynamic_trace_bindings,
    )

    document = json.loads(receipt_path.read_text())
    artifact = next(
        item
        for item in document["receipt"]["artifacts"]
        if item["role"] == "dynamic_trace"
    )
    path = root / artifact["path"]
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    trace = trace_pb2.Trace()
    trace.ParseFromString(path.read_bytes())
    mutate(trace)
    payload = trace.SerializeToString()
    path.write_bytes(payload)
    artifact["sha256"] = admit.sha256_bytes(payload)
    artifact["size_bytes"] = len(payload)
    _rewrite(receipt_path, document)


def _append_nvbit_auxiliary_artifacts(
    root: Path, receipt_path: Path, payloads: list[bytes],
) -> None:
    document = json.loads(receipt_path.read_text())
    receipt = document["receipt"]
    slug = receipt["identity"]["slug"]
    for index, payload in enumerate(payloads):
        relative = f"nvbit/{slug}/auxiliary_{index}.bin"
        path = root / relative
        path.write_bytes(payload)
        receipt["artifacts"].append({
            "role": "trace_auxiliary",
            "path": relative,
            "sha256": admit.sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    _rewrite(receipt_path, document)


def test_preflight_content_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(tmp_path, target, "preflight", lambda value: value.update({"unexpected": True}))
    with pytest.raises(admit.AdmissionError, match="preflight content"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_ncu_manifest_must_equal_closed_handoff_contract(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(
        tmp_path,
        target,
        "label_run_manifest",
        lambda value: value.pop("kernel_signature_source"),
    )
    with pytest.raises(admit.AdmissionError, match="closed successor contract"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_ncu_manifest_duplicate_keys_are_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    artifact = next(
        item for item in document["receipt"]["artifacts"]
        if item["role"] == "label_run_manifest"
    )
    manifest = tmp_path / artifact["path"]
    payload = manifest.read_text().replace(
        '"source_commit":',
        f'"source_commit": "{admit.PINNED_BUNDLE_COMMIT}",\n  "source_commit":',
        1,
    ).encode()
    manifest.write_bytes(payload)
    artifact["sha256"] = admit.sha256_bytes(payload)
    artifact["size_bytes"] = len(payload)
    _rewrite(target, document)

    with pytest.raises(admit.AdmissionError, match="duplicate JSON key"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("profiler_family", "other_profiler"),
        ("replay_mode", "application"),
        ("warmup_count", 7),
        ("evidence_class", "calibration"),
    ),
)
def test_ncu_sidecar_producer_pins_are_rederived(
    tmp_path: Path,
    field: str,
    replacement: object,
):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(
        tmp_path,
        target,
        "label_sidecar",
        lambda value: value.update({field: replacement}),
    )

    with pytest.raises(admit.AdmissionError, match="sidecar identity"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_ncu_sidecar_variant_axis_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(
        tmp_path,
        target,
        "label_sidecar",
        lambda value: value.update({
            "register_cap": 64,
            "variant_id": "register_cap_64",
        }),
    )

    with pytest.raises(admit.AdmissionError, match="capture producer"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_ncu_artifacts_require_canonical_launch_set_paths(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    artifact = next(
        item for item in document["receipt"]["artifacts"]
        if item["role"] == "ncu_report"
    )
    source = tmp_path / artifact["path"]
    relocated = tmp_path / "relocated.ncu-rep"
    source.replace(relocated)
    artifact["path"] = relocated.relative_to(tmp_path).as_posix()
    _rewrite(target, document)

    with pytest.raises(admit.AdmissionError, match="canonical launch-set namespace"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_environment_retired_driver_authority_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(
        tmp_path,
        target,
        "environment",
        lambda value: value["live_checks"].update({
            "driver_version": "610.43.02",
        }),
    )

    with pytest.raises(admit.AdmissionError, match="live-check pins are invalid"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_environment_preflight_binding_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(tmp_path, target, "environment", lambda value: value.update({"preflight_pins_sha256": "0" * 64}))
    with pytest.raises(admit.AdmissionError, match="environment pin binding"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_unavailable_boot_identity_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    process = document["receipt"]["process"]
    process["boot_id_sha256"] = "0" * 64
    process.pop("process_identity_sha256")
    process["process_identity_sha256"] = admit.canonical_sha(process)
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="boot identity is unavailable"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_environment_live_clock_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(tmp_path, target, "environment", lambda value: value["live_checks"].update({"sm_clock_mhz": 1410}))
    with pytest.raises(admit.AdmissionError, match="live-check pins"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_environment_live_output_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(tmp_path, target, "environment", lambda value: value["live_outputs"].update({"smi_stdout": "forged\n"}))
    with pytest.raises(admit.AdmissionError, match="live output hash mismatch"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_environment_ncu_authority_path_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)

    def mutate(value: dict) -> None:
        authority = value["live_checks"]["ncu_authority"]
        authority["resolved_path"] = str(
            Path(authority["launcher_resolved_path"]).parent
            / "target/other/ncu"
        )

    _mutate_json_artifact(tmp_path, target, "environment", mutate)
    with pytest.raises(admit.AdmissionError, match="authority paths differ"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_environment_ncu_authority_hash_tamper_is_rejected(tmp_path: Path):
    for index, field in enumerate(("launcher_sha256", "executable_sha256")):
        root = tmp_path / str(index)
        target = _receipt(root, MATRIX["rows"][0], "ncu", 0)

        def mutate(value: dict, *, field: str = field) -> None:
            value["live_checks"]["ncu_authority"][field] = "0" * 64

        _mutate_json_artifact(root, target, "environment", mutate)
        with pytest.raises(admit.AdmissionError, match="executable pins differ"):
            admit.build_overlay(
                root, generated_at_utc="2026-07-23T01:00:00Z"
            )


def test_environment_ncu_launcher_path_pin_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)

    def mutate(value: dict) -> None:
        authority = value["live_checks"]["ncu_authority"]
        launcher = Path("/opt/forged/ncu")
        authority["launcher_resolved_path"] = str(launcher)
        authority["resolved_path"] = str(
            launcher.parent / authority["launcher_target_relpath"]
        )

    _mutate_json_artifact(tmp_path, target, "environment", mutate)
    with pytest.raises(admit.AdmissionError, match="executable pins differ"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_environment_ncu_authority_version_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)

    def mutate(value: dict) -> None:
        value["live_checks"]["ncu_authority"]["version_line"] = (
            "Version 1900.0.0 (build forged)"
        )

    _mutate_json_artifact(tmp_path, target, "environment", mutate)
    with pytest.raises(admit.AdmissionError, match="version line differs"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_ncu_command_authority_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    receipt = document["receipt"]
    index = receipt["command"].index("--ncu-executable-sha256") + 1
    receipt["command"][index] = "c" * 64
    command_sha256 = admit.canonical_sha(receipt["command"])
    receipt["command_sha256"] = command_sha256
    receipt["process"]["command_sha256"] = command_sha256
    receipt["process"].pop("process_identity_sha256")
    receipt["process"]["process_identity_sha256"] = admit.canonical_sha(
        receipt["process"]
    )
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="command authority differs"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_ncu_executor_provenance_tamper_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    document["receipt"]["provenance"]["action_executor_sha256"] = "0" * 64
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="tool provenance"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_ncu_schema_requires_executor_provenance(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    document["receipt"]["provenance"].pop("action_executor_sha256")
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="schema validation failed"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_nvbit_schema_forbids_executor_provenance(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    document = json.loads(target.read_text())
    document["receipt"]["provenance"]["action_executor_sha256"] = (
        admit.NCU_ACTION_EXECUTOR_SHA256
    )
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="schema validation failed"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_ncu_non_authority_command_drift_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    receipt = document["receipt"]
    receipt["command"][receipt["command"].index("--warmup") + 1] = "1"
    command_sha256 = admit.canonical_sha(receipt["command"])
    receipt["command_sha256"] = command_sha256
    receipt["process"]["command_sha256"] = command_sha256
    receipt["process"].pop("process_identity_sha256")
    receipt["process"]["process_identity_sha256"] = admit.canonical_sha(
        receipt["process"]
    )
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="exact row"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_nvbit_command_drift_is_rejected(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    document = json.loads(target.read_text())
    receipt = document["receipt"]
    receipt["command"] = [*receipt["command"], "--forged"]
    command_sha256 = admit.canonical_sha(receipt["command"])
    receipt["command_sha256"] = command_sha256
    receipt["process"]["command_sha256"] = command_sha256
    receipt["process"].pop("process_identity_sha256")
    receipt["process"]["process_identity_sha256"] = admit.canonical_sha(
        receipt["process"]
    )
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="NVBit command authority"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_receipt_command_digest_is_recomputed(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    receipt = document["receipt"]
    receipt["command_sha256"] = "0" * 64
    receipt["process"]["command_sha256"] = "0" * 64
    receipt["process"].pop("process_identity_sha256")
    receipt["process"]["process_identity_sha256"] = admit.canonical_sha(
        receipt["process"]
    )
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="command digest"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_ncu_version_output_must_equal_exact_authority(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)

    def mutate(value: dict) -> None:
        forged = "unrelated output containing 2026.1.0\n"
        value["live_outputs"]["ncu_stdout"] = forged
        value["live_checks"]["ncu_stdout_sha256"] = admit.sha256_bytes(
            forged.encode()
        )

    _mutate_json_artifact(tmp_path, target, "environment", mutate)
    with pytest.raises(admit.AdmissionError, match="exact authority"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_ncu_environment_rejects_nvbit_injection_controls(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(tmp_path, target, "environment", lambda value: value["environment"].update({"CUDA_INJECTION64_PATH": "/tmp/tracer_tool.so"}))
    with pytest.raises(admit.AdmissionError, match="environment names"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_ncu_environment_rejects_nonproducer_executor_keys(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _mutate_json_artifact(
        tmp_path,
        target,
        "environment",
        lambda value: value["environment"].update({"HOME": "/tmp/forged"}),
    )
    with pytest.raises(admit.AdmissionError, match="environment names"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_nvbit_environment_rejects_ambient_loader_controls(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    _mutate_json_artifact(
        tmp_path,
        target,
        "environment",
        lambda value: value["environment"].update({"LD_PRELOAD": "/tmp/hostile.so"}),
    )
    with pytest.raises(admit.AdmissionError, match="NVBit collection environment"):
        admit.build_overlay(
            tmp_path, generated_at_utc="2026-07-23T01:00:00Z"
        )


def test_ncu_sidecar_duplicate_required_metric_is_rejected(tmp_path: Path, monkeypatch):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    sidecar = _report_sidecar()
    sidecar["unit_tagged_metrics"].append(copy.deepcopy(sidecar["unit_tagged_metrics"][0]))
    observation = _report_observation()
    observation["report_sha256"] = admit.sha256_bytes(report.read_bytes())
    monkeypatch.setattr(
        admit,
        "_import_ncu_report_csv",
        lambda _identity, _path: (
            _REPORT_CSV,
            observation["report_sha256"],
            None,
        ),
    )
    with pytest.raises(admit.AdmissionError, match="duplicate unit-tagged"):
        admit._verify_ncu_report(
            _expected_ncu_identity(),
            report,
            observation["report_sha256"],
            sidecar,
            observation,
            *_report_launch_authority(),
        )


def test_new_schemas_are_closed_draft_2020_12():
    for path in (RECEIPT_SCHEMA, OVERLAY_SCHEMA):
        schema = json.loads(path.read_text())
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"].endswith("draft/2020-12/schema")
        assert schema["additionalProperties"] is False


def test_legacy_v1_receipt_cannot_enter_v2_admission(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    document = json.loads(target.read_text())
    document["schema_version"] = 1
    _rewrite(target, document)

    with pytest.raises(admit.AdmissionError, match="schema validation failed"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_valid_joined_pairs_produce_capture_ready_chunk(tmp_path: Path):
    _all_pairs(tmp_path)
    overlay = admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")
    assert overlay["counts"] == {"total": 3, "pending": 0, "capture_ready": 3, "chunks": 1}
    assert all(row["status"] == "capture_ready" for row in overlay["rows"])
    assert len(overlay["chunks"]) == 1
    Draft202012Validator(json.loads(OVERLAY_SCHEMA.read_text()), format_checker=None).validate(overlay)


def test_receipt_generation_replacement_before_inventory_is_rejected(
    tmp_path: Path, monkeypatch,
):
    _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    original = admit._receipt_ref
    replaced = False

    def replace_after_reference(generation, root, relative, seal):
        nonlocal replaced
        reference = original(generation, root, relative, seal)
        if not replaced:
            replaced = True
            path = root / relative
            replacement = path.with_name(f"{path.name}.replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return reference

    monkeypatch.setattr(admit, "_receipt_ref", replace_after_reference)
    with pytest.raises(admit.AdmissionError, match="immutable generation validation"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_overlay_publication_remains_inside_retained_receipt_generation(
    tmp_path: Path, monkeypatch,
):
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    receipt = _receipt(receipt_root, MATRIX["rows"][0], "ncu", 0)
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output = output_parent / "overlay.json"
    original = admit.immutable.publish_regular_file

    def publish_then_mutate(*args, after_commit=None, **kwargs):
        assert after_commit is not None

        def mutate_then_authorize():
            replacement = receipt.with_name(f"{receipt.name}.replacement")
            replacement.write_bytes(receipt.read_bytes())
            replacement.replace(receipt)
            after_commit()

        return original(
            *args,
            after_commit=mutate_then_authorize,
            **kwargs,
        )

    monkeypatch.setattr(
        admit.immutable,
        "publish_regular_file",
        publish_then_mutate,
    )

    with pytest.raises(admit.AdmissionError, match="immutable generation validation"):
        admit.build_and_write_overlay(
            receipt_root,
            output,
            generated_at_utc="2026-07-23T01:00:00Z",
        )

    assert not output.exists()
    retired = [
        path
        for path in output_parent.iterdir()
        if path.name.startswith(".failed-regular-")
        and not path.name.endswith(".salvaging")
    ]
    assert len(retired) == 1
    assert json.loads(retired[0].read_bytes())["overlay_seal_sha256"]


def test_overlay_writer_rejects_replaced_parent_without_redirecting(
    tmp_path: Path, monkeypatch,
):
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    moved_parent = tmp_path / "moved-output"
    attacker_parent = tmp_path / "attacker"
    attacker_parent.mkdir()
    output = output_parent / "overlay.json"
    original = admit.immutable.write_regular_bytes_at

    def replace_parent(parent_descriptor, name, payload, *, mode):
        output_parent.rename(moved_parent)
        output_parent.symlink_to(attacker_parent, target_is_directory=True)
        original(parent_descriptor, name, payload, mode=mode)

    monkeypatch.setattr(
        admit.immutable,
        "write_regular_bytes_at",
        replace_parent,
    )

    with pytest.raises(admit.AdmissionError, match="exclusively write overlay"):
        admit.write_overlay({"draft": True}, output, receipt_root)

    assert not (attacker_parent / output.name).exists()
    assert (moved_parent / output.name).is_file()


def test_overlay_writer_rejects_symlinked_parent_chain(tmp_path: Path):
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(admit.AdmissionError, match="exclusively write overlay"):
        admit.write_overlay(
            {"draft": True},
            alias / "overlay.json",
            receipt_root,
        )

    assert not (real_parent / "overlay.json").exists()


def test_ncu_launch_observation_requires_closed_presentation_contract(
    tmp_path: Path,
):
    target = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)

    def mutate(observation: dict) -> None:
        observation["kernel_name_base"] = "demangled"
        observation["observation_sha256"] = admit.canonical_sha({
            key: value
            for key, value in observation.items()
            if key != "observation_sha256"
        })

    _mutate_json_artifact(
        tmp_path, target, "ncu_launch_observation", mutate,
    )
    with pytest.raises(admit.AdmissionError, match="observation contract"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("producer", "forged_tracer"),
        ("binary_version", 87),
        ("nvbit_version", "1.9"),
    ],
)
def test_nvbit_trace_summary_producer_metadata_is_pinned(
    tmp_path: Path, field: str, value,
):
    target = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    _mutate_json_artifact(
        tmp_path,
        target,
        "trace_summary",
        lambda summary: summary.update({field: value}),
    )
    with pytest.raises(admit.AdmissionError, match="trace summary semantics"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("binary_version", 87),
        ("nvbit_version", "1.9"),
    ],
)
def test_coordinated_nvbit_producer_version_drift_is_rejected(
    tmp_path: Path, field: str, value,
):
    target = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    _mutate_dynamic_trace_artifact(
        tmp_path,
        target,
        lambda trace: setattr(trace, field, value),
    )
    _mutate_json_artifact(
        tmp_path,
        target,
        "trace_summary",
        lambda summary: summary.update({field: value}),
    )
    with pytest.raises(admit.AdmissionError, match="producer version metadata"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_multiple_nvbit_auxiliary_artifacts_are_admitted(tmp_path: Path):
    row = MATRIX["rows"][0]
    _receipt(tmp_path, row, "ncu", 0)
    target = _receipt(tmp_path, row, "nvbit", 0)
    _append_nvbit_auxiliary_artifacts(
        tmp_path, target, [b"first auxiliary", b"second auxiliary"],
    )

    overlay = admit.build_overlay(
        tmp_path,
        generated_at_utc="2026-07-23T01:00:00Z",
    )
    assert overlay["rows"][0]["status"] == "capture_ready"


def test_nvbit_typed_stats_must_share_the_validated_trace_root(tmp_path: Path):
    row = MATRIX["rows"][0]
    target = _receipt(tmp_path, row, "nvbit", 0)
    document = json.loads(target.read_text())
    receipt = document["receipt"]
    actual_stats = next(
        artifact
        for artifact in receipt["artifacts"]
        if artifact["role"] == "stats"
    )
    actual_stats["role"] = "trace_auxiliary"
    forged_path = tmp_path / "nvbit" / "forged" / "stats.csv"
    forged_path.parent.mkdir(parents=True)
    forged_payload = b"forged stats"
    forged_path.write_bytes(forged_payload)
    receipt["artifacts"].append({
        "role": "stats",
        "path": forged_path.relative_to(tmp_path).as_posix(),
        "sha256": admit.sha256_bytes(forged_payload),
        "size_bytes": len(forged_payload),
    })
    _rewrite(target, document)

    with pytest.raises(admit.AdmissionError, match="outside canonical trace root"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_admission_rechecks_immutable_nvbit_trace_byte_budget(
    tmp_path: Path, monkeypatch,
):
    _all_pairs(tmp_path)
    target = tmp_path / f"00_nvbit{admit.RECEIPT_SUFFIX}"
    _append_nvbit_auxiliary_artifacts(
        tmp_path, target, [b"a" * 17, b"b" * 19],
    )
    receipt = json.loads(target.read_text())["receipt"]
    trace_roles = {
        "dynamic_trace",
        "stats",
        "threadblock",
        "trace_summary",
        "trace_auxiliary",
    }
    trace_sizes = [
        artifact["size_bytes"]
        for artifact in receipt["artifacts"]
        if artifact["role"] in trace_roles
    ]
    aggregate_budget = max(trace_sizes) + 1
    assert aggregate_budget < sum(trace_sizes)
    matrix = copy.deepcopy(MATRIX)
    matrix["rows"][0]["trace_budget"]["max_trace_bytes"] = aggregate_budget
    monkeypatch.setattr(
        admit,
        "validate_immutable_bundle",
        lambda: (None, None, copy.deepcopy(ROSTER), matrix),
    )

    with pytest.raises(admit.AdmissionError, match="trace byte budget"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_ncu_nvbit_observed_kernel_mismatch_is_rejected(tmp_path: Path):
    row = MATRIX["rows"][0]
    _receipt(tmp_path, row, "ncu", 0)
    target = _receipt(tmp_path, row, "nvbit", 0)
    document = json.loads(target.read_text())
    artifact = next(
        item
        for item in document["receipt"]["artifacts"]
        if item["role"] == "dynamic_trace"
    )
    trace_path = tmp_path / artifact["path"]
    from tools.nvbit_tracer.proto_binding_loader import (
        load_dynamic_trace_bindings,
    )

    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    trace = trace_pb2.Trace()
    trace.ParseFromString(trace_path.read_bytes())
    trace.gpu_device[0].streams[0].kernels[0].name = (
        "_Zdifferent_kernel___0"
    )
    payload = trace.SerializeToString()
    trace_path.write_bytes(payload)
    artifact["sha256"] = admit.sha256_bytes(payload)
    artifact["size_bytes"] = len(payload)
    _rewrite(target, document)

    with pytest.raises(admit.AdmissionError, match="observed launch join"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_absent_receipts_are_typed_pending_and_have_zero_chunks(tmp_path: Path):
    overlay = admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")
    assert overlay["dispatch"]["typed_reason"] == "capture_receipts_absent"
    assert overlay["counts"] == {"total": 3, "pending": 3, "capture_ready": 0, "chunks": 0}
    assert overlay["chunks"] == []
    assert {row["typed_reason"] for row in overlay["rows"]} == {"ncu_and_nvbit_receipts_absent"}


def test_one_missing_variant_stays_pending(tmp_path: Path):
    row = MATRIX["rows"][0]
    _receipt(tmp_path, row, "ncu", 0)
    overlay = admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")
    assert overlay["rows"][0]["typed_reason"] == "nvbit_receipt_absent"
    assert overlay["rows"][0]["status"] == "pending"
    assert overlay["counts"]["chunks"] == 0


@pytest.mark.parametrize("case", ["hash", "identity", "launch", "unknown", "duplicate", "extra"])
def test_receipt_validation_rejects_negative_cases(tmp_path: Path, case: str):
    _all_pairs(tmp_path)
    target = tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}"
    document = json.loads(target.read_text())
    if case == "hash":
        report = next(
            artifact for artifact in document["receipt"]["artifacts"]
            if artifact["role"] == "ncu_report"
        )
        (tmp_path / report["path"]).write_bytes(b"drift")
    elif case == "identity":
        document["receipt"]["identity"]["arch"] = "sm_120"
        _rewrite(target, document)
    elif case == "launch":
        document["receipt"]["launch_identity"]["grid_dim"][0] += 1
        _rewrite(target, document)
    elif case == "unknown":
        document["receipt"]["unexpected"] = True
        _rewrite(target, document)
    elif case == "duplicate":
        shutil.copy2(target, tmp_path / f"duplicate_ncu{admit.RECEIPT_SUFFIX}")
    elif case == "extra":
        (tmp_path / "unreferenced.bin").write_bytes(b"extra")
    with pytest.raises(admit.AdmissionError):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_receipt_path_escape_is_rejected_by_schema(tmp_path: Path):
    row = MATRIX["rows"][0]
    path = _receipt(tmp_path, row, "ncu", 0)
    document = json.loads(path.read_text())
    document["receipt"]["artifacts"][0]["path"] = "../escape.bin"
    _rewrite(path, document)
    with pytest.raises(admit.AdmissionError):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_symlink_in_external_receipt_inventory_is_rejected(tmp_path: Path):
    _all_pairs(tmp_path)
    receipt = json.loads(
        (tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}").read_text()
    )["receipt"]
    report = next(
        artifact for artifact in receipt["artifacts"]
        if artifact["role"] == "ncu_report"
    )
    link = tmp_path / "symlink.bin"
    link.symlink_to(tmp_path / report["path"])
    with pytest.raises(admit.AdmissionError, match="symlink"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_same_process_receipts_cannot_promote(tmp_path: Path):
    _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    nvbit = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    document = json.loads(nvbit.read_text())
    ncu = json.loads((tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}").read_text())
    process = document["receipt"]["process"]
    process["pid"] = ncu["receipt"]["process"]["pid"]
    process["started_at_utc"] = ncu["receipt"]["process"]["started_at_utc"]
    process.pop("process_identity_sha256")
    process["process_identity_sha256"] = admit.canonical_sha(process)
    _rewrite(nvbit, document)
    with pytest.raises(admit.AdmissionError, match="separate processes"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def _replace_receipt_gpu_uuid(
    root: Path,
    receipt_path: Path,
    gpu_uuid: str,
) -> None:
    document = json.loads(receipt_path.read_text())
    receipt = document["receipt"]
    environment_artifact = next(
        artifact for artifact in receipt["artifacts"]
        if artifact["role"] == "environment"
    )
    environment_path = root / environment_artifact["path"]
    environment = json.loads(environment_path.read_text())
    identity_stdout = f"{gpu_uuid}, 610.43.03\n"
    environment["live_outputs"]["identity_stdout"] = identity_stdout
    environment["live_checks"]["identity_stdout_sha256"] = admit.sha256_bytes(
        identity_stdout.encode()
    )
    environment["live_checks"]["gpu_uuid_sha256"] = admit.sha256_bytes(
        gpu_uuid.encode()
    )
    encoded = json.dumps(environment, sort_keys=True).encode()
    environment_path.write_bytes(encoded)
    environment_artifact["sha256"] = admit.sha256_bytes(encoded)
    environment_artifact["size_bytes"] = len(encoded)
    receipt["environment_sha256"] = environment_artifact["sha256"]
    process = receipt["process"]
    process["host_identity_sha256"] = admit.canonical_sha({
        "gpu": environment["live_checks"]["gpu_name"],
        "smi": environment["live_checks"]["smi_stdout_sha256"],
        "gpu_uuid_sha256": environment["live_checks"]["gpu_uuid_sha256"],
        "driver_version": environment["live_checks"]["driver_version"],
        "identity_stdout_sha256": environment["live_checks"][
            "identity_stdout_sha256"
        ],
    })
    process.pop("process_identity_sha256")
    process["process_identity_sha256"] = admit.canonical_sha(process)
    _rewrite(receipt_path, document)


def test_different_gpu_uuid_receipts_cannot_join(tmp_path: Path):
    _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    nvbit = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    _replace_receipt_gpu_uuid(tmp_path, nvbit, OTHER_GPU_UUID)

    with pytest.raises(admit.AdmissionError, match="same host boot"):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


@pytest.mark.parametrize(
    "malformed_uuid",
    (
        "GPU-11111111111111111111111111111111",
        "GPU-11111111-1111-1111-1111-11111111111",
        "GPU-11111111-1111-1111-1111-111111111111-",
    ),
)
def test_malformed_gpu_uuid_is_rejected(
    tmp_path: Path,
    malformed_uuid: str,
):
    ncu = _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    _replace_receipt_gpu_uuid(tmp_path, ncu, malformed_uuid)

    with pytest.raises(
        admit.AdmissionError,
        match="selected-GPU identity output is not pinned",
    ):
        admit.build_overlay(
            tmp_path,
            generated_at_utc="2026-07-23T01:00:00Z",
        )


def test_extra_receipt_kind_is_rejected(tmp_path: Path):
    _all_pairs(tmp_path)
    target = tmp_path / f"00_nvbit{admit.RECEIPT_SUFFIX}"
    document = json.loads(target.read_text())
    document["receipt"]["capture_kind"] = "ncu"
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_process_command_hash_must_bind_receipt_command(tmp_path: Path):
    _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    target = tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}"
    document = json.loads(target.read_text())
    document["receipt"]["command"] = ["tampered"]
    document["receipt"]["command_sha256"] = admit.canonical_sha(["tampered"])
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="process command"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_kernel_name_must_bind_successor_identity(tmp_path: Path):
    _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    target = tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}"
    document = json.loads(target.read_text())
    document["receipt"]["launch_identity"]["kernel_name"] = "wrong_successor"
    launch = document["receipt"]["launch_identity"]
    launch["launch_identity_sha256"] = admit.canonical_sha(
        {key: value for key, value in launch.items() if key != "launch_identity_sha256"}
    )
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="kernel_name"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_nvbit_receipt_requires_stats_and_threadblock_payloads(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    document = json.loads(target.read_text())
    document["receipt"]["artifacts"] = [
        artifact
        for artifact in document["receipt"]["artifacts"]
        if artifact["role"] not in {"stats", "threadblock"}
    ]
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")
