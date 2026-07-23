"""C1 ownership tests for the tracked NVBit Python protobuf authorities."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

import nvbit_trace_summary_adapter as adapter  # noqa: E402
import run_suite  # noqa: E402


def test_kernel_signature_matches_every_committed_golden_fixture() -> None:
    fixture_root = SUITE_ROOT / "golden" / "kernel_signature_fixtures"
    fixtures = sorted(path for path in fixture_root.iterdir() if path.is_dir())
    assert fixtures
    for fixture in fixtures:
        inputs = json.loads((fixture / "inputs.json").read_text(encoding="utf-8"))
        expected_canonical = (fixture / "expected_canonical.txt").read_text(
            encoding="utf-8"
        ).removesuffix("\n")
        expected_sha = (fixture / "expected_sha256.txt").read_text(
            encoding="utf-8"
        ).strip()
        assert adapter.canonical_serialize(inputs) == expected_canonical, fixture
        assert adapter.compute_kernel_signature_sha256(inputs) == expected_sha, fixture


def test_missing_tracked_binding_is_repository_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter,
        "_TRACKED_BINDING_NAMES",
        (*adapter._TRACKED_BINDING_NAMES, "__planted_missing_pb2.py"),
    )
    with pytest.raises(adapter.NvbitTrackedProtobufCorruption) as caught:
        adapter._import_pb_modules()
    diagnostic = str(caught.value)
    assert "tracked protobuf authority" in diagnostic
    assert "_pb_generated/__planted_missing_pb2.py" in diagnostic
    assert "git restore --source=HEAD --worktree --" in diagnostic
    assert "do not regenerate" in diagnostic


def test_missing_protobuf_runtime_is_prerequisite_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_runtime(name: str, *args, **kwargs):
        if name == "google.protobuf" or name.startswith("google.protobuf."):
            raise ImportError("planted missing runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_runtime)
    with pytest.raises(adapter.NvbitProtobufRuntimePrerequisiteMissing) as caught:
        adapter._import_pb_modules()
    diagnostic = str(caught.value)
    assert "runtime prerequisite is missing" in diagnostic
    assert "requirements.txt" in diagnostic


def test_run_suite_dependency_failures_propagate_hard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def reject_adapter(name: str, *args, **kwargs):
        if name == "nvbit_trace_summary_adapter":
            raise ImportError("planted missing tracked adapter")
        return original_import(name, *args, **kwargs)

    cases: list[tuple[str, object]] = [("adapter_import_failed", reject_adapter)]

    def corrupt_enumeration(_path: Path) -> int:
        raise adapter.NvbitTrackedProtobufCorruption("planted corruption")

    fake_adapter = SimpleNamespace(
        enumerate_launches=corrupt_enumeration,
        NvbitTrackedProtobufCorruption=adapter.NvbitTrackedProtobufCorruption,
        NvbitProtobufRuntimePrerequisiteMissing=(
            adapter.NvbitProtobufRuntimePrerequisiteMissing
        ),
    )
    cases.append(("tracked_protobuf_corrupt", fake_adapter))

    for expected_status, planted in cases:
        with monkeypatch.context() as context:
            if callable(planted):
                context.setattr(builtins, "__import__", planted)
            else:
                context.setitem(sys.modules, "nvbit_trace_summary_adapter", planted)
            result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
                traces_folder=tmp_path,
                trace_pb=tmp_path / "dynamic_trace.pb",
                slug="fixture",
                shape_point="fixture_shape",
                executable=tmp_path / "fixture_run",
                tracer_so_path=tmp_path / "tracer_tool.so",
                artifact_dir=tmp_path,
            )
        assert result["status"] == expected_status
        assert result["failed_hard"] is True
        assert result["paths"] == []
        assert "skipped" not in result["status"]


def test_all_active_consumers_enforce_tracked_ownership() -> None:
    consumers = (
        REPO_ROOT / "tools/nvbit_tracer/import_trace_contract.py",
        REPO_ROOT / "tools/nvbit_tracer/trace_admission.py",
        SUITE_ROOT / "nvbit_trace_summary_adapter.py",
        REPO_ROOT / "tools/cache_predictor/l1_production_helper.py",
        REPO_ROOT
        / "tools/sm_86_predictor_accuracy_recovery_v2"
        / "compute_warp_pool_l1_per_admitted_cell.py",
        REPO_ROOT / "tools/local_labels/extend_launch_parameters.py",
        SUITE_ROOT / "run_suite.py",
    )
    forbidden = (
        "nvbit-pb-python",
        "skipped_adapter_import_failed",
        "skipped_missing_protobuf_bindings",
        'typed_reason="protobuf_bindings_missing"',
        "degrade gracefully",
    )
    for path in consumers:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, (path, token)
    assert "TraceContractTrackedBindingsCorrupt" in consumers[0].read_text()
    assert "tracked_protobuf_corrupt" in consumers[1].read_text()
    assert "NvbitTrackedProtobufCorruption" in consumers[2].read_text()
    assert "L1ProductionHelperTrackedBindingsCorrupt" in consumers[3].read_text()
    assert "hard exceptions propagate" in consumers[4].read_text()
    assert "LaunchParametersTrackedBindingsCorrupt" in consumers[5].read_text()
    assert "protobuf_runtime_prerequisite_missing" in consumers[6].read_text()
