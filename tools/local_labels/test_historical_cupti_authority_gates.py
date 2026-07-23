"""Closed gates for retained historical CUPTI replay authority."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SIMBENCH = _REPO_ROOT / "workloads" / "simbench_bw"
_SCHEMAS = _SIMBENCH / "schemas"
_HISTORICAL_CUPTI = _SCHEMAS / "historical" / "cupti_activity_sidecar.v1.schema.json"
_CUPTI_FIXTURE = (
    _SIMBENCH
    / "golden"
    / "cupti_nvbit_join_fixtures"
    / "medium_rmsnorm_bf16"
    / "cupti_sidecar.json"
)
_DUAL_RUN_FIXTURE = (
    _SIMBENCH
    / "tests"
    / "fixtures"
    / "simbench_bw_validation"
    / "passing"
    / "dual_run_manifest__rmsnorm.json"
)
_DUAL_RUN_SCHEMA = _SCHEMAS / "dual_run_manifest.schema.json"
_HISTORICAL_CUPTI_SHA256 = (
    "836c0db4f6f5a2a582a9b334be02c829b40a094f709259d8008f05f200faaeb4"
)
_CUPTI_FIXTURE_SHA256 = (
    "4cbcfa03339b39a7f9972f7be84c5510686f406d272ec0d16e11db3e6202e05a"
)
_DUAL_RUN_FIXTURE_SHA256 = (
    "d51107272e04b735f38fbb791bf566a7a08eb4261ab4d667ff06607e7234497a"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_VALIDATE = _load_module("_historical_cupti_validate", _SIMBENCH / "validate.py")
_DUAL_RUN_CONSUMER = _load_module(
    "_historical_cupti_dual_run_consumer",
    _SIMBENCH / "dual_run_manifest_consumer.py",
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _tracked_worktree_files() -> set[Path]:
    payload = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=_REPO_ROOT
    )
    paths = {
        _REPO_ROOT / raw.decode("utf-8")
        for raw in payload.split(b"\0")
        if raw
    }
    # The migration target is deliberately visible to this gate before it is
    # staged as well as in the eventual linked staged-tree and HEAD checks.
    paths.add(_HISTORICAL_CUPTI)
    return {path for path in paths if path.is_file() and not path.is_symlink()}


def test_g10_old_cupti_basename_map_is_exact() -> None:
    old_basename = "cupti_activity_" + "sidecar.schema.json"
    token = old_basename.encode("ascii")
    assert Path(__file__).read_bytes().count(token) == 0

    mentions: Counter[str] = Counter()
    for path in _tracked_worktree_files():
        count = path.read_bytes().count(token)
        if count:
            mentions[path.relative_to(_REPO_ROOT).as_posix()] = count

    assert mentions == Counter(
        {
            "plans/cache_predictor_first_geomean_microbench_label_overlap_plan.md": 1,
            "plans/l1_baseline_l2_corpus_expansion_plan.md": 3,
            "plans/research/sm86_wallclock_prediction_research_dossier.md": 1,
            "plans/sm120_tier_expansion_wallclock_cache_metrics_plan.md": 3,
            "plans/sm86_nonroot_wallclock_prediction_plan.md": 1,
            "workloads/simbench_bw/schemas/historical/"
            "cupti_activity_sidecar.v1.schema.json": 1,
            "workloads/simbench_bw/schemas/historical/README.md": 1,
            "data/sm_86_predictor_accuracy_recovery_v2/"
            "ac6_missing_feature_candidates.json": 1,
        }
    )
    assert len(mentions) == 8
    assert sum(mentions.values()) == 12


def test_g11_golden_cupti_sidecar_replays_against_historical_schema() -> None:
    assert hashlib.sha256(_HISTORICAL_CUPTI.read_bytes()).hexdigest() == (
        _HISTORICAL_CUPTI_SHA256
    )
    assert hashlib.sha256(_CUPTI_FIXTURE.read_bytes()).hexdigest() == (
        _CUPTI_FIXTURE_SHA256
    )
    jsonschema.validate(
        instance=_load_json(_CUPTI_FIXTURE),
        schema=_load_json(_HISTORICAL_CUPTI),
    )
    assert _VALIDATE.validate_cupti_activity_sidecar(_CUPTI_FIXTURE) == []


def test_g12_frozen_dual_run_manifest_exceeds_drift() -> None:
    assert hashlib.sha256(_DUAL_RUN_FIXTURE.read_bytes()).hexdigest() == (
        _DUAL_RUN_FIXTURE_SHA256
    )
    manifest = _load_json(_DUAL_RUN_FIXTURE)
    jsonschema.validate(instance=manifest, schema=_load_json(_DUAL_RUN_SCHEMA))
    with pytest.raises(_DUAL_RUN_CONSUMER.DualRunDriftExceeded) as captured:
        _DUAL_RUN_CONSUMER.admit_dual_run_manifest(manifest)
    assert captured.value.drift_observed_ns == 300_000_000_000
    assert captured.value.drift_tolerance_ns == 5_000_000


def test_g12b_boundary_drift_is_admitted_without_fixture_write() -> None:
    before = _DUAL_RUN_FIXTURE.read_bytes()
    manifest = copy.deepcopy(json.loads(before))
    for launch in manifest["launches"]:
        cupti_timestamp = launch["cupti_run_nvml_state"][
            "sample_timestamp_ns_host_monotonic"
        ]
        launch["nvbit_run_nvml_state"][
            "sample_timestamp_ns_host_monotonic"
        ] = cupti_timestamp + manifest["drift_tolerance_ns"]
    jsonschema.validate(instance=manifest, schema=_load_json(_DUAL_RUN_SCHEMA))
    _DUAL_RUN_CONSUMER.admit_dual_run_manifest(manifest)
    assert _DUAL_RUN_FIXTURE.read_bytes() == before


def test_g13_live_cupti_production_surfaces_are_retired() -> None:
    retired = (
        "tools/build/run_cloud_arm_evidence.sh",
        "tools/build/run_local_dual_source_lcd.py",
        "tools/build/verify_cloud_arm_evidence.py",
        "tools/local_labels/capture_timing_deploy_ns_sidecars.py",
        "tools/local_labels/collect_portable_lcd_duration.py",
        "tools/local_labels/duration_repeatability_aggregator.py",
        "tools/local_labels/duration_trace_goldilocks_sweep.py",
        "workloads/simbench_bw/calibrate_dual_source_tolerance.py",
        "workloads/simbench_bw/cupti_activity_collector.cu",
        "workloads/simbench_bw/cupti_activity_collector.h",
        "workloads/simbench_bw/cupti_dual_source_live_admission.py",
    )
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=_REPO_ROOT
    ).split(b"\0")
    tracked_names = {raw.decode("utf-8") for raw in tracked if raw}
    for relative in retired:
        assert relative not in tracked_names
        assert not (_REPO_ROOT / relative).exists()

    forbidden_cli = ("--enable-cupti-activity", "cupti-collector-lib")
    for relative in tracked_names:
        path = _REPO_ROOT / relative
        if not path.is_file() or path.name.startswith("test_"):
            continue
        if not (
            path.name == "Makefile"
            or path.suffix in {".mk", ".py", ".sh"}
        ):
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden_cli:
            assert token not in source, (relative, token)


def test_g14_consumers_resolve_historical_schema_without_live_producer() -> None:
    consumers = (
        _SIMBENCH / "validate.py",
        _SIMBENCH / "merge_evidence.py",
        _SIMBENCH / "dual_run_manifest_emitter.py",
        _REPO_ROOT
        / "tools"
        / "sm_86_predictor_accuracy_recovery_v2"
        / "build_missing_feature_candidates.py",
    )
    old_basename = "cupti_activity_" + "sidecar.schema.json"
    new_basename = "cupti_activity_sidecar.v1.schema.json"
    subprocess_calls = {"call", "check_call", "check_output", "Popen", "run"}
    for path in consumers:
        source = path.read_text(encoding="utf-8")
        assert old_basename not in source
        assert "historical" in source
        assert new_basename in source
        tree = ast.parse(source, filename=str(path))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in subprocess_calls
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            for node in ast.walk(tree)
        )
