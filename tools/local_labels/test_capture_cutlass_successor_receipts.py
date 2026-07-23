from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.local_labels import capture_cutlass_successor_receipts as capture


MATRIX_KEY = "cutlass_successor:vendor_gemm_cublas:vendor_gemm_cutlass"
VARIANT_ID = "vendor_gemm_cutlass_bf16_simt_trace00_v1"


def _pinned_preflight(tmp_path: Path) -> Path:
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    document = capture._expected_preflight(row, roster_row)
    path = tmp_path / "successor_preflight.json"
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    return path


def test_preflight_clock_refusal_happens_before_live_gpu_queries(tmp_path: Path, monkeypatch):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["sm_clock_mhz"] = 1410
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda: pytest.fail("live GPU query must not run after static preflight refusal"),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *args, **kwargs: pytest.fail("capture process must not run"),
    )
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="ncu",
            preflight_path=path,
            output_root=tmp_path / "capture-output",
        )
    assert not list((tmp_path / "capture-output").glob("*.capture_receipt.json"))


def test_preflight_requires_exact_successor_identity(tmp_path: Path):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["binary_sha256"] = "0" * 64
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.validate_successor_preflight(path, row, roster_row)


def test_preflight_rejects_extras_and_wrong_toolchain(tmp_path: Path):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["cuda_release"] = {"major": 13, "minor": 3}
    document["unexpected"] = True
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.validate_successor_preflight(path, row, roster_row)


def test_preflight_rejects_non_absolute_input(tmp_path: Path):
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="absolute"):
        capture.validate_successor_preflight(Path("relative-preflight.json"), row, roster_row)


def test_preexisting_sibling_files_are_allowed_but_current_output_is_safe(tmp_path: Path):
    root = tmp_path / "output"
    root.mkdir()
    (root / "prior-phase").mkdir()
    (root / "prior-phase" / "prior.ncu-rep").write_bytes(b"prior")
    current = root / "nvbit" / "successor"
    capture._reject_preexisting_output_descendants(
        root, output_dir=current, reserved_files=(root / "preflight_nvbit.json",)
    )

    current.parent.mkdir(parents=True)
    current.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(capture.CaptureError, match="unsafe"):
        capture._reject_preexisting_output_descendants(
            root, output_dir=current, reserved_files=()
        )


def test_preexisting_output_files_are_rejected(tmp_path: Path):
    root = tmp_path / "output-file"
    root.mkdir()
    output_dir = root / "ncu"
    output_dir.write_bytes(b"stale")
    with pytest.raises(capture.CaptureError, match="unsafe"):
        capture._reject_preexisting_output_descendants(
            root, output_dir=output_dir, reserved_files=()
        )


def test_zero_size_artifacts_are_rejected(tmp_path: Path):
    root = tmp_path / "output"
    root.mkdir()
    empty = root / "empty.pb"
    empty.touch()
    with pytest.raises(capture.CaptureError, match="zero-sized"):
        capture._artifact(root, empty, "threadblock")


def test_trace_producer_version_mismatch_is_rejected(tmp_path: Path):
    trace_pb2 = capture._import_trace_pb2()
    trace = trace_pb2.Trace()
    trace.trace_contract_id = capture.EXPECTED_TRACE_CONTRACT_ID
    trace.memory_payload_classes.extend(capture.EXPECTED_MEMORY_PAYLOAD_CLASSES)
    trace.binary_version = 120
    trace.nvbit_version = capture.NVBIT_VERSION
    (tmp_path / "dynamic_trace.pb").write_bytes(trace.SerializeToString())
    with pytest.raises(capture.CaptureError, match="producer version mismatch"):
        capture._validate_trace_producer(tmp_path)


def test_capture_timeout_is_bounded(tmp_path: Path):
    with pytest.raises(capture.CaptureError, match="timeout"):
        capture._run_capture_command(
            ["python", "-c", "import time; time.sleep(2)"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=0.05,
        )


def test_trace_byte_budget_stops_capture(tmp_path: Path):
    command = [
        capture.sys.executable,
        "-c",
        f"from pathlib import Path; import time; Path({str(tmp_path / 'payload.bin')!r}).write_bytes(b'x' * 64); time.sleep(2)",
    ]
    with pytest.raises(capture.CaptureError, match="byte budget"):
        capture._run_capture_command(
            command,
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=8,
        )


def test_environment_provenance_names_trace_folder_controls():
    document = capture._environment_document(
        {"pin": "static"}, {"check": "live"},
        {"TRACES_FOLDER": "/capture/traces", "USER_DEFINED_FOLDERS": "1"},
    )
    assert document["environment"]["TRACES_FOLDER"] == "/capture/traces"
    assert document["environment"]["USER_DEFINED_FOLDERS"] == "1"
