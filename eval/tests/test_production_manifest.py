from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from types import MappingProxyType

import pytest
from jsonschema import Draft202012Validator

from eval import production_manifest as pm
from eval.contracts._common import (
    AuthorityBlockerCode,
    EvaluatorAuthorityUnavailable,
)
from eval.contracts.launch_crosswalk import (
    FixtureV2ExpectedLaunch,
    FixtureV2ParserAuthority,
    LaunchNamespace,
    create_workload_execution_authority_fixture,
)
from eval.contracts.model_seal import RetainedPayloadSnapshot
from tools.local_labels.immutable_artifact import (
    canonical_json_bytes,
    seal_model,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
CANARY = ROOT / "sim/run/tests/fixtures/addr_carry_canary_bundle"
EXPECTED_IDENTITY = "78d5ec89090badad0477d2b9ed329ed69baa03b7b537afed3e16845108704367"
WORKLOAD_EXECUTABLE = "e" * 64
SIMULATOR_EXECUTABLE = "f" * 64


def _write_canonical(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    return path


def _install_clean_card_projections(bundle: Path) -> None:
    arch = json.loads((bundle / "cards/sm_86.json").read_text())
    latencies = arch["latencies_cycles"]
    arch_projection = {
        "schema_version": 1,
        "artifact_kind": "target_free_sim_arch_card_projection",
        "arch_id": arch["arch_id"],
        "sku": arch["sku"],
        "compute_capability": arch["compute_capability"],
        "build_arch_family": arch["build_arch_family"],
        "sm_count": arch["sm_count"],
        "l1_geometry": {
            name: arch["l1_geometry"][name]
            for name in (
                "l1tex_bytes",
                "ways",
                "default_l1_bytes",
                "max_shared_bytes",
                "sets",
            )
        },
        "l2_geometry": {
            **{
                name: arch["l2_geometry"][name]
                for name in (
                    "capacity_bytes",
                    "sector_bytes",
                    "num_slices",
                    "line_bytes",
                    "policy",
                )
            },
            "provenance": "fixture_arch_authority_v1:sha256:"
            + sha256_bytes((bundle / "cards/sm_86.json").read_bytes()),
        },
        "nvml_reference_operating_point": {
            name: arch["nvml_reference_operating_point"][name]
            for name in (
                "sm_clock_ref_mhz",
                "max_sm_clock_mhz",
                "short_burst_sm_clock_ref_mhz",
            )
        },
        "dram_bandwidth": {
            **{
                name: arch["dram_bandwidth"][name]
                for name in (
                    "peak_bytes_per_second",
                    "locked_sm_clock_mhz",
                    "locked_mem_clock_mhz",
                )
            },
            "provenance": "fixture_arch_authority_v1:sha256:"
            + sha256_bytes((bundle / "cards/sm_86.json").read_bytes()),
        },
        "latencies_cycles": {
            **{
                name: {
                    key: latencies[name][key]
                    for key in ("value", "dispersion", "unit")
                }
                for name in ("l1_hit", "l2_hit", "dram", "shared")
            },
            "mshr_outstanding_misses": {
                "status": "research_gap",
                "reason": "fixture_authority_gap",
                "challenge": "fixture_pchase_authority_v1",
            },
            "provenance": "fixture_pchase_authority_v1:sha256:"
            + sha256_bytes(
                (bundle / "cards/sm_86_static_card.json").read_bytes()
            ),
        },
        "bare_ldg_e_default": arch["bare_ldg_e_default"],
    }
    arch_payload = canonical_json_bytes(arch_projection)
    (bundle / "cards/sm_86.json").write_bytes(arch_payload)

    pchase = json.loads((bundle / "cards/sm_86_static_card.json").read_text())
    property_names = (
        "l1_capacity_bytes",
        "l1_hit_latency_cycles",
        "l2_hit_latency_cycles",
        "mshr_outstanding_misses",
        "l1_replacement_policy",
        "l1_associativity_ways",
    )
    pchase_projection = {
        "schema_version": 1,
        "artifact_kind": "target_free_sim_pchase_card_projection",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "device": "NVIDIA GeForce RTX 3090 Ti",
        "build": pchase["build"],
        "properties": {
            name: {
                key: pchase["properties"][name][key]
                for key in ("kind", "value", "typed_reason", "unit")
                if key in pchase["properties"][name]
            }
            for name in property_names
        },
    }
    pchase_payload = canonical_json_bytes(pchase_projection)
    (bundle / "cards/sm_86_static_card.json").write_bytes(pchase_payload)

    manifest_path = bundle / "launch_evidence.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact, payload in (
        ("arch_card", arch_payload),
        ("pchase_card", pchase_payload),
    ):
        manifest["artifacts"][artifact]["size_bytes"] = len(payload)
        manifest["artifacts"][artifact]["sha256"] = sha256_bytes(payload)
    _write_json(manifest_path, manifest)


def _copy_clean_bundle(destination: Path) -> Path:
    shutil.copytree(CANARY, destination)
    _install_clean_card_projections(destination)
    return destination


def _parser_authorities() -> dict[LaunchNamespace, FixtureV2ParserAuthority]:
    return {
        namespace: FixtureV2ParserAuthority(
            namespace=namespace,
            authority_id=f"fixture-{namespace.value.lower()}-parser",
            authority_sha256=sha256_bytes(namespace.value.encode()),
        )
        for namespace in LaunchNamespace
    }


def _workload_model(
    *,
    launch_count: int = 1,
    argv: tuple[str, ...] = ("canary",),
    shape: tuple[str, ...] = ("n=32",),
    kernel_family: str = "canary",
) -> dict:
    schema_bytes, schema_snapshot = RetainedPayloadSnapshot.read_path(
        pm.WORKLOAD_AUTHORITY_SCHEMA_PATH
    )
    launches = tuple(
        FixtureV2ExpectedLaunch(
            process_launch_ordinal=index,
            kernel_mangled_name="_Z17addr_carry_canaryPKiPiS1_S1_i___0",
            grid=(1, 1, 1),
            block=(32, 1, 1),
            device_id=0,
            stream_id=0,
            binary_architecture="sm_86",
        )
        for index in range(launch_count)
    )
    return create_workload_execution_authority_fixture(
        executable_sha256=WORKLOAD_EXECUTABLE,
        argv=argv,
        shape=shape,
        binary_architecture="sm_86",
        gpu_sku="rtx_3090_ti",
        kernel_family=kernel_family,
        expected_launches=launches,
        execution_nonce="fixture-execution",
        parser_authorities=_parser_authorities(),
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
    )


_WORKLOAD_BOUNDARY_CASES = (
    (
        "colon-posix",
        "--input:/home/alice/private.bin",
        "canary",
        "absolute",
    ),
    ("colon-home", "--input:~/private.bin", "canary", "absolute"),
    (
        "colon-drive",
        r"--input:C:\private\private.bin",
        "canary",
        "absolute",
    ),
    (
        "colon-unc",
        r"--input:\\server\share\private.bin",
        "canary",
        "absolute",
    ),
    (
        "colon-uri",
        "--input:file://host/private.bin",
        "canary",
        "absolute",
    ),
    (
        "colon-traversal",
        "--input:../private.bin",
        "canary",
        "traversal",
    ),
    (
        "nested-colon-posix",
        "--namespace:input:/home/alice/private.bin",
        "canary",
        "absolute",
    ),
    (
        "double-colon-posix",
        "--input::/home/alice/private.bin",
        "canary",
        "absolute",
    ),
    (
        "double-colon-home",
        "--input::~/private.bin",
        "canary",
        "absolute",
    ),
    (
        "double-colon-drive",
        "--input::C:/private/private.bin",
        "canary",
        "absolute",
    ),
    (
        "double-colon-traversal",
        "--input::../private.bin",
        "canary",
        "traversal",
    ),
    (
        "deep-colon-posix",
        "--namespace:input:format:binary:/home/alice/private.bin",
        "canary",
        "absolute",
    ),
    (
        "max-colon-delimiters",
        ":" * 131_071,
        "NCUReport",
        "forbidden",
    ),
    (
        "max-equals-delimiters",
        "=" * 131_071,
        "NCUReport",
        "forbidden",
    ),
    ("scalar-alias", None, "NCUReport", "forbidden"),
)


def _boundary_workload_model(argument: str | None, kernel_family: str) -> dict:
    argv = ("canary",) if argument is None else ("canary", argument)
    return _workload_model(argv=argv, kernel_family=kernel_family)


def _build_model(*, workload_executable: str = WORKLOAD_EXECUTABLE) -> dict:
    return seal_model(
        {
            "schema_version": 1,
            "artifact_kind": pm.FIXTURE_BUILD_KIND,
            "status": pm.FIXTURE_CONTRACT_MODE,
            "build_id": "fixture-build",
            "source_tree_sha256": "a" * 64,
            "source_commit": "b" * 40,
            "traced_workload_executable_sha256": workload_executable,
            "simulator_executable_sha256": SIMULATOR_EXECUTABLE,
            "arch_id": "sm_86",
            "sku": "rtx_3090_ti",
        }
    )


def _admission_model(workload: dict, bundle: Path) -> dict:
    replay = {
        "kernel_mangled_name": "_Z17addr_carry_canaryPKiPiS1_S1_i___0",
        "grid": {"x": 1, "y": 1, "z": 1},
        "block": {"x": 32, "y": 1, "z": 1},
        "device_id": 0,
        "stream_id": 0,
        "binary_architecture": "sm_86",
    }
    core = {
        "schema_version": 1,
        "artifact_kind": pm.FIXTURE_ADMISSION_KIND,
        "authority_status": pm.FIXTURE_CONTRACT_MODE,
        "existing_admission_id": "canary-rust-admission",
        "source_content_manifest_id": "canary-source-generation",
        "source_content_manifest_sha256": "c" * 64,
        "workload_execution_authority_model_seal_sha256": workload[
            "model_seal_sha256"
        ],
        "launch_evidence_manifest_sha256": sha256_bytes(
            (bundle / "launch_evidence.json").read_bytes()
        ),
        "launch_evidence_identity_sha256": EXPECTED_IDENTITY,
        "replay_authenticated_launch": replay,
        "workload_execution_executable_sha256": WORKLOAD_EXECUTABLE,
        "architecture": {"arch_id": "sm_86", "sku": "rtx_3090_ti"},
    }


    core.update(
        {
            "control_bits_sha256": sha256_bytes(
                (bundle / "control/canary_sm86.control_bits.json").read_bytes()
            ),
            "arch_card_sha256": sha256_bytes(
                (bundle / "cards/sm_86.json").read_bytes()
            ),
            "pchase_card_sha256": sha256_bytes(
                (bundle / "cards/sm_86_static_card.json").read_bytes()
            ),
            "coverage": pm.EXACT_COVERAGE,
            "binary_digest_authority": pm.NO_BINARY_DIGEST_AUTHORITY,
        }
    )
    return seal_model(core)


def _artifact_paths(bundle: Path) -> dict[str, Path]:
    paths = {
        "trace/dynamic_trace.pb": bundle / "trace/dynamic_trace.pb",
        "trace/extra_info/allocations.json": bundle
        / "trace/extra_info/allocations.json",
        "trace/stats.csv": bundle / "trace/stats.csv",
        "trace/extra_info/enhanced_execution_info.json": bundle
        / "trace/extra_info/enhanced_execution_info.json",
        "control/canary_sm86.control_bits.json": bundle
        / "control/canary_sm86.control_bits.json",
        "cards/sm_86.json": bundle / "cards/sm_86.json",
        "cards/sm_86_static_card.json": bundle / "cards/sm_86_static_card.json",
    }
    for payload in (bundle / "trace/threadblocks").rglob("*.pb"):
        relative = payload.relative_to(bundle).as_posix()
        paths[relative] = payload
    return paths


def _prepare(
    tmp_path: Path,
    *,
    workload_model: dict | None = None,
) -> tuple[
    pm.FixtureProductionManifestBuild,
    pm.FixtureRetainedLaunchEvidence,
    pm.FixtureWorkloadExecutionAuthority,
    pm.FixtureDeclaredBuildIdentity,
    Path,
]:
    bundle = tmp_path / "source-bundle"
    _copy_clean_bundle(bundle)
    inputs = tmp_path / "source-authority"
    if workload_model is None:
        workload_model = _workload_model()
    workload_path = _write_canonical(inputs / "workload.json", workload_model)
    build_path = _write_canonical(inputs / "build.json", _build_model())
    admission_path = _write_canonical(
        inputs / "existing-admission.json",
        _admission_model(workload_model, bundle),
    )
    workload = pm.load_fixture_workload_execution_authority(workload_path)
    declared_build = pm.load_fixture_declared_build_identity(build_path)
    admission = pm.load_fixture_existing_launch_admission_projection(
        admission_path.read_bytes()
    )
    evidence = pm.retain_fixture_launch_evidence(
        bundle / "launch_evidence.json",
        destination_manifest_path="launches/launch-000/launch_evidence.json",
        artifact_paths=_artifact_paths(bundle),
        admission=admission,
    )
    build = pm.create_sim_run_manifest_fixture(
        generation_id="fixture-generation",
        scope="fixture-canary-v1",
        workload_authority=workload,
        launch_evidence=(evidence,),
        replay_mode={"mode": "flat"},
        build_identity=declared_build,
    )
    return build, evidence, workload, declared_build, bundle


def _unchecked_retained_workload(
    tmp_path: Path,
    model: dict,
) -> pm.FixtureWorkloadExecutionAuthority:
    source = pm.retain_target_free_source_fixture(
        _write_canonical(tmp_path / "unchecked-workload.json", model),
        source_id="fixture.unchecked-workload",
        role=pm.SourceRole.WORKLOAD_AUTHORITY,
        relative_path="authority_inputs/workload_execution_authority.v1.json",
        maximum_bytes=pm.MAX_MANIFEST_BYTES,
    )
    schema = pm._schema_retained_source(
        pm.WORKLOAD_AUTHORITY_SCHEMA_PATH,
        name="workload-authority",
        schema_id=pm.WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID,
        expected_sha256=pm.WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256,
    )
    return pm.FixtureWorkloadExecutionAuthority(
        model,
        source,
        schema,
        _construction_token=pm._PRIVATE_WORKLOAD_TOKEN,
    )


def test_real_production_is_typed_unavailable_before_any_path_open(monkeypatch):
    registries = (
        pm.PRODUCTION_WORKLOAD_EXECUTION_AUTHORITIES,
        pm.PRODUCTION_EXISTING_LAUNCH_ADMISSIONS,
        pm.PRODUCTION_SIMULATOR_BUILD_AUTHORITIES,
        pm.PRODUCTION_MANIFEST_PUBLICATION_AUTHORITIES,
    )
    assert all(isinstance(item, MappingProxyType) and not item for item in registries)
    opens: list[object] = []

    def forbidden_open(*args, **kwargs):
        opens.append((args, kwargs))
        raise AssertionError("production unavailable gate opened a path")

    monkeypatch.setattr(pm, "read_regular_snapshot", forbidden_open)
    monkeypatch.setattr(os, "open", forbidden_open)
    for entry in (
        pm.create_production_sim_run_manifest,
        pm.publish_production_sim_run_manifest,
        pm.load_production_sim_run_manifest_authority,
    ):
        with pytest.raises(EvaluatorAuthorityUnavailable) as caught:
            entry(
                ncu_report_path="/forbidden/report.ncu-rep",
                matched_labels_path="/forbidden/matched-labels.json",
                baseline_runtime_path="/forbidden/baseline.sqlite",
            )
        assert caught.value.code is AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE
    assert opens == []


def test_fixture_construction_is_deterministic_full_and_target_free(tmp_path: Path):
    build, evidence, workload, declared_build, _bundle = _prepare(tmp_path)
    pm.verify_sim_run_manifest_fixture(build)
    second = pm.create_sim_run_manifest_fixture(
        generation_id="fixture-generation",
        scope="fixture-canary-v1",
        workload_authority=workload,
        launch_evidence=(evidence,),
        replay_mode={"mode": "flat"},
        build_identity=declared_build,
    )
    assert build.manifest_bytes == second.manifest_bytes
    assert [
        (item.relative_path, item.sha256, item.size_bytes)
        for item in build.generation_files
    ] == [
        (item.relative_path, item.sha256, item.size_bytes)
        for item in second.generation_files
    ]
    manifest = build.manifest_document
    assert manifest["schema_version"] == 2
    assert manifest["architecture"] == {"arch_id": "sm_86", "sku": "rtx_3090_ti"}
    assert manifest["launches"][0]["expected_identity_sha256"] == EXPECTED_IDENTITY
    assert set(manifest["launches"][0]["launch_key"]) == {
        "workload_id",
        "matched_execution_id",
        "process_launch_ordinal",
        "kernel_mangled_name",
        "grid",
        "block",
        "device_id",
        "stream_id",
        "binary_architecture",
    }
    assert evidence.admission.document["authority_status"] == pm.FIXTURE_CONTRACT_MODE
    assert evidence.manifest_document["binary_digest"]["status"] == "typed_gap"


def test_fixture_publication_seals_generation_and_detached_authority(tmp_path: Path):
    build, _evidence, _workload, _declared_build, _bundle = _prepare(tmp_path)
    published = pm.publish_sim_run_manifest_fixture(build, tmp_path / "published")
    loaded = pm.load_sim_run_manifest_authority_fixture(
        published.authority_path,
        published.generation_path,
    )
    assert loaded == published.authority_document
    assert loaded["contract_mode"] == pm.FIXTURE_CONTRACT_MODE
    assert loaded["schema_sha256"] == pm.AUTHORITY_SCHEMA_SHA256
    assert loaded["manifest_identity"]["sha256"] == sha256_bytes(
        build.manifest_bytes
    )
    assert loaded["generation_identity"]["content_manifest_sha256"] == (
        published.generation.content_manifest_sha256
    )


    assert loaded["producer_identity"]["producer_sha256"] == sha256_bytes(
        Path(pm.__file__).read_bytes()
    )
    assert stat.S_IMODE(published.generation_path.stat().st_mode) == 0o555
    assert stat.S_IMODE(published.authority_path.stat().st_mode) == 0o444
    assert published.authority_path.stat().st_nlink == 1
    serialized = published.authority_bytes.decode()
    for forbidden in (
        "unit_tagged_metrics",
        "matched_labels",
        "accuracy_report",
        "baseline_runtime",
        "same_cell_calibration",
    ):
        assert forbidden not in serialized


def _retain_bundle(tmp_path: Path, bundle: Path) -> pm.FixtureRetainedLaunchEvidence:
    workload = _workload_model()
    admission = pm.load_fixture_existing_launch_admission_projection(
        canonical_json_bytes(_admission_model(workload, bundle))
    )
    return pm.retain_fixture_launch_evidence(
        bundle / "launch_evidence.json",
        destination_manifest_path="launches/launch-000/launch_evidence.json",
        artifact_paths=_artifact_paths(bundle),
        admission=admission,
    )


def test_forbidden_source_paths_are_rejected_before_open(tmp_path: Path, monkeypatch):
    safe = tmp_path / "safe.json"
    safe.write_text("{}", encoding="utf-8")
    opened: list[Path] = []

    def spy(path, **kwargs):
        opened.append(Path(path))
        raise AssertionError("forbidden path reached the opener")

    monkeypatch.setattr(pm, "read_regular_snapshot", spy)
    for name in (
        "ncu",
        "NCUReport",
        "label",
        "local_labels",
        "matchedLabel",
        "target_values",
        "accuracy",
        "baseline",
        "runtime-record",
        "same-cell-calibration",
    ):
        with pytest.raises(pm.ProductionManifestError, match="forbidden"):
            pm.retain_target_free_source_fixture(
                tmp_path / name / "payload.json",
                source_id="fixture-safe-source",
                role=pm.SourceRole.BUILD_MANIFEST,
                relative_path="authority_inputs/safe.json",
            )
    assert opened == []


@pytest.mark.parametrize(
    ("_case_name", "argument", "kernel_family", "message"),
    _WORKLOAD_BOUNDARY_CASES,
    ids=[case[0] for case in _WORKLOAD_BOUNDARY_CASES],
)
def test_workload_boundary_rejects_at_initial_load(
    tmp_path: Path,
    _case_name: str,
    argument: str | None,
    kernel_family: str,
    message: str,
):
    model = _boundary_workload_model(argument, kernel_family)
    path = _write_canonical(tmp_path / "workload.json", model)
    with pytest.raises(pm.ProductionManifestError, match=message) as caught:
        pm.load_fixture_workload_execution_authority(path)
    assert caught.value.code == "PRODUCTION_MANIFEST_INVALID"


@pytest.mark.parametrize(
    ("_case_name", "argument", "kernel_family", "message"),
    _WORKLOAD_BOUNDARY_CASES,
    ids=[case[0] for case in _WORKLOAD_BOUNDARY_CASES],
)
def test_workload_boundary_rejects_at_retained_recheck(
    tmp_path: Path,
    _case_name: str,
    argument: str | None,
    kernel_family: str,
    message: str,
):
    model = _boundary_workload_model(argument, kernel_family)
    retained = _unchecked_retained_workload(tmp_path, model)
    with pytest.raises(pm.ProductionManifestError, match=message) as caught:
        retained.recheck()
    assert caught.value.code == "PRODUCTION_MANIFEST_INVALID"


@pytest.mark.parametrize(
    ("_case_name", "argument", "kernel_family", "message"),
    _WORKLOAD_BOUNDARY_CASES,
    ids=[case[0] for case in _WORKLOAD_BOUNDARY_CASES],
)
def test_workload_boundary_rejects_at_detached_generation_load(
    tmp_path: Path,
    monkeypatch,
    _case_name: str,
    argument: str | None,
    kernel_family: str,
    message: str,
):
    model = _boundary_workload_model(argument, kernel_family)
    validator = pm._validate_workload_target_free_model
    monkeypatch.setattr(pm, "_validate_workload_target_free_model", lambda _value: None)
    build, *_ = _prepare(tmp_path / "staged", workload_model=model)
    published = pm.publish_sim_run_manifest_fixture(build, tmp_path / "published")
    monkeypatch.setattr(pm, "_validate_workload_target_free_model", validator)

    with pytest.raises(pm.ProductionManifestError, match=message) as caught:
        pm.load_sim_run_manifest_authority_fixture(
            published.authority_path,
            published.generation_path,
        )
    assert caught.value.code == "PRODUCTION_MANIFEST_INVALID"


def test_workload_boundary_preserves_benign_scalar_metadata_and_arguments(
    tmp_path: Path,
):
    model = _workload_model(
        argv=(
            "canary",
            "--mode:concurrency",
            "--input:data/private.bin",
            "--namespace:input:data/private.bin",
            "--input::data/private.bin",
            "--format:binary:little-endian",
        ),
        shape=("n=32", "layout:row-major"),
        kernel_family="concurrency",
    )
    retained = pm.load_fixture_workload_execution_authority(
        _write_canonical(tmp_path / "workload.json", model)
    )
    assert retained.recheck() == model


@pytest.mark.parametrize(
    "value",
    (
        "NCUReport",
        "MatchedLabels",
        "targetValues",
        "target_value=12345",
    ),
)
def test_workload_argv_rejects_normalized_target_assignments(
    tmp_path: Path, value: str
):
    model = _workload_model()
    model["argv"].append(value)
    _reseal_in_place(model)
    path = _write_canonical(tmp_path / "workload.json", model)
    with pytest.raises(pm.ProductionManifestError, match="forbidden"):
        pm.load_fixture_workload_execution_authority(path)


def test_workload_shape_rejects_target_assignment(tmp_path: Path):
    model = _workload_model()
    model["shape"].append("target_value=12345")
    _reseal_in_place(model)
    path = _write_canonical(tmp_path / "workload.json", model)
    with pytest.raises(pm.ProductionManifestError, match="forbidden"):
        pm.load_fixture_workload_execution_authority(path)


@pytest.mark.parametrize("key", ("targetValues", "ncuReport"))
def test_workload_rejects_nested_normalized_target_keys(
    tmp_path: Path, key: str
):
    model = _workload_model()
    model["fixtureExtension"] = {"nested": {key: "forbidden"}}
    _reseal_in_place(model)
    path = _write_canonical(tmp_path / "workload.json", model)
    with pytest.raises(pm.ProductionManifestError, match="forbidden field"):
        pm.load_fixture_workload_execution_authority(path)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("/home/alice/private/input.json", "absolute"),
        ("~/private/input.json", "home"),
        (r"C:\\private\\input.json", "drive"),
        (r"\\\\server\\share\\input.json", "UNC"),
        ("file://host/input.json", "URI"),
        ("../private/input.json", "traversal"),
    ),
)
def test_workload_rejects_noncanonical_input_paths(
    tmp_path: Path, value: str, message: str
):
    model = _workload_model()
    model["argv"].append(value)
    _reseal_in_place(model)
    path = _write_canonical(tmp_path / "workload.json", model)
    with pytest.raises(pm.ProductionManifestError, match=message):
        pm.load_fixture_workload_execution_authority(path)


def test_workload_rejects_nested_traversal_regardless_of_key_name(tmp_path: Path):
    model = _workload_model()
    model["fixtureExtension"] = {"opaque": "../private/input.json"}
    _reseal_in_place(model)
    path = _write_canonical(tmp_path / "workload.json", model)
    with pytest.raises(pm.ProductionManifestError, match="traversal"):
        pm.load_fixture_workload_execution_authority(path)


def test_retained_inputs_reject_symlinks_hardlinks_and_path_drift(tmp_path: Path):
    symlink_bundle = tmp_path / "symlink-bundle"
    _copy_clean_bundle(symlink_bundle)
    dynamic = symlink_bundle / "trace/dynamic_trace.pb"
    original = dynamic.read_bytes()
    dynamic.unlink()
    dynamic.symlink_to(symlink_bundle / "trace/stats.csv")
    with pytest.raises(pm.ProductionManifestError):
        _retain_bundle(tmp_path, symlink_bundle)

    hardlink_bundle = tmp_path / "hardlink-bundle"
    _copy_clean_bundle(hardlink_bundle)
    dynamic = hardlink_bundle / "trace/dynamic_trace.pb"
    dynamic.unlink()
    os.link(hardlink_bundle / "trace/stats.csv", dynamic)
    with pytest.raises(pm.ProductionManifestError):
        _retain_bundle(tmp_path, hardlink_bundle)

    build, _evidence, _workload, _declared, bundle = _prepare(tmp_path / "drift-case")
    (bundle / "trace/stats.csv").write_bytes(b"drift\n")
    with pytest.raises(pm.ProductionManifestError, match="drift"):
        build.recheck()
    assert original


def _reseal_in_place(model: dict) -> None:
    model.pop("model_seal_sha256", None)
    sealed = seal_model(model)
    model.clear()
    model.update(sealed)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_unsupported_sampled_and_cross_arch_launch_evidence_are_rejected(tmp_path: Path):
    unsupported = tmp_path / "unsupported-bundle"
    _copy_clean_bundle(unsupported)
    manifest = json.loads((unsupported / "launch_evidence.json").read_text())
    manifest["schema_version"] = 3
    _write_json(unsupported / "launch_evidence.json", manifest)
    with pytest.raises(pm.ProductionManifestError, match="schema validation|unsupported"):
        _retain_bundle(tmp_path, unsupported)

    sampled = tmp_path / "sampled-bundle"
    _copy_clean_bundle(sampled)
    sample_path = sampled / "trace/trace_sampling.json"
    sample_path.write_bytes(b"{}")
    manifest = json.loads((sampled / "launch_evidence.json").read_text())
    manifest["artifacts"]["trace_sampling_json"] = {
        "status": "present",
        "artifact": {
            "path": "trace/trace_sampling.json",
            "size_bytes": 2,
            "sha256": sha256_bytes(b"{}"),
        },
    }
    _write_json(sampled / "launch_evidence.json", manifest)
    with pytest.raises(pm.ProductionManifestError, match="sampled|sampling"):
        _retain_bundle(tmp_path, sampled)

    cross_arch = tmp_path / "cross-arch-bundle"
    _copy_clean_bundle(cross_arch)
    manifest = json.loads((cross_arch / "launch_evidence.json").read_text())
    manifest["architecture"] = {"arch_id": "sm_120", "sku": "rtx_5060_laptop"}
    _write_json(cross_arch / "launch_evidence.json", manifest)
    with pytest.raises(pm.ProductionManifestError, match="architecture"):
        _retain_bundle(tmp_path, cross_arch)


def _refresh_artifact_claim(bundle: Path, artifact_name: str, relative: str) -> None:
    manifest_path = bundle / "launch_evidence.json"
    manifest = json.loads(manifest_path.read_text())
    payload = (bundle / relative).read_bytes()
    manifest["artifacts"][artifact_name]["size_bytes"] = len(payload)
    manifest["artifacts"][artifact_name]["sha256"] = sha256_bytes(payload)
    _write_json(manifest_path, manifest)


def test_control_and_required_card_research_gaps_are_rejected(tmp_path: Path):
    arch_gap = tmp_path / "arch-gap-bundle"
    _copy_clean_bundle(arch_gap)
    card_path = arch_gap / "cards/sm_86.json"
    card = json.loads(card_path.read_text())
    card["latencies_cycles"]["l1_hit"] = {
        "status": "research_gap",
        "reason": "fixture unresolved",
    }
    _write_json(card_path, card)
    _refresh_artifact_claim(arch_gap, "arch_card", "cards/sm_86.json")
    with pytest.raises(pm.ProductionManifestError) as caught:
        _retain_bundle(tmp_path, arch_gap)
    assert caught.value.code == "CARD_PROJECTION_IDENTITY_MISMATCH"

    pchase_gap = tmp_path / "pchase-gap-bundle"
    _copy_clean_bundle(pchase_gap)
    card_path = pchase_gap / "cards/sm_86_static_card.json"
    card = json.loads(card_path.read_text())
    card["properties"]["l1_capacity_bytes"] = {
        "kind": "typed",
        "typed_reason": "geometry_unresolved",
    }
    _write_json(card_path, card)
    _refresh_artifact_claim(
        pchase_gap,
        "pchase_card",
        "cards/sm_86_static_card.json",
    )
    with pytest.raises(pm.ProductionManifestError) as caught:
        _retain_bundle(tmp_path, pchase_gap)
    assert caught.value.code == "CARD_PROJECTION_IDENTITY_MISMATCH"

    control_gap = tmp_path / "control-gap-bundle"
    _copy_clean_bundle(control_gap)
    sidecar_path = control_gap / "control/canary_sm86.control_bits.json"
    sidecar = json.loads(sidecar_path.read_text())
    first_function = next(iter(sidecar["functions"].values()))
    first_function[0]["status"] = "research_gap"
    _write_json(sidecar_path, sidecar)
    _refresh_artifact_claim(
        control_gap,
        "control_bits_sidecar",
        "control/canary_sm86.control_bits.json",
    )
    with pytest.raises(pm.ProductionManifestError, match="control-bit gap"):
        _retain_bundle(tmp_path, control_gap)


def test_partial_duplicate_and_executable_mismatch_fail_closed(tmp_path: Path):
    build, evidence, workload, declared, _bundle = _prepare(tmp_path)
    with pytest.raises(pm.ProductionManifestError, match="nonempty complete"):
        pm.create_sim_run_manifest_fixture(
            generation_id="empty-generation",
            scope="fixture-canary-v1",
            workload_authority=workload,
            launch_evidence=(),
            replay_mode={"mode": "flat"},
            build_identity=declared,
        )
    with pytest.raises(pm.ProductionManifestError, match="cardinality"):
        pm.create_sim_run_manifest_fixture(
            generation_id="duplicate-generation",
            scope="fixture-canary-v1",
            workload_authority=workload,
            launch_evidence=(evidence, evidence),
            replay_mode={"mode": "flat"},
            build_identity=declared,
        )

    bad_path = _write_canonical(
        tmp_path / "source-authority/bad-build.json",
        _build_model(workload_executable="d" * 64),
    )
    bad_build = pm.load_fixture_declared_build_identity(bad_path)
    with pytest.raises(pm.ProductionManifestError, match="executable hashes disagree"):
        pm.create_sim_run_manifest_fixture(
            generation_id="bad-build-generation",
            scope="fixture-canary-v1",
            workload_authority=workload,
            launch_evidence=(evidence,),
            replay_mode={"mode": "flat"},
            build_identity=bad_build,
        )
    assert build.manifest_document["launches"]


def test_fixture_builder_has_no_ambient_git_fallback(tmp_path: Path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ambient process/Git fallback invoked")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    build, *_rest = _prepare(tmp_path)
    assert build.build_identity["source_commit"] == "b" * 40


def test_caller_cannot_construct_retained_source_or_mutate_registries(tmp_path: Path):
    payload = b"{}"
    metadata = (1, 2, stat.S_IFREG | 0o644, 1, len(payload), 3, 4)
    with pytest.raises(pm.ProductionManifestError, match="retention API"):
        pm.RetainedTargetFreeSource(
            source_id="fixture-source",
            role=pm.SourceRole.BUILD_MANIFEST,
            relative_path="authority_inputs/source.json",
            payload=payload,
            sha256=sha256_bytes(payload),
            size_bytes=len(payload),
            metadata=metadata,
            _source_path=tmp_path / "source.json",
        )
    with pytest.raises(TypeError):
        pm.PRODUCTION_EXISTING_LAUNCH_ADMISSIONS["caller"] = object()


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "labels",
        "unit_tagged_metrics",
        "accuracy_report",
        "baseline_runtime",
        "same_cell_calibration",
        "predictions",
    ],
)
def test_target_and_baseline_fields_cannot_enter_declared_build(
    tmp_path: Path, forbidden_key: str
):
    model = _build_model()
    model.pop("model_seal_sha256")
    model[forbidden_key] = {"forbidden": True}
    path = _write_canonical(tmp_path / f"build-{forbidden_key}.json", seal_model(model))
    with pytest.raises(pm.ProductionManifestError, match="fields are not exact|forbidden"):
        pm.load_fixture_declared_build_identity(path)


def test_scope_cannot_launder_baseline_or_calibration_identity(tmp_path: Path):
    _build, evidence, workload, declared, _bundle = _prepare(tmp_path)
    for scope in ("baseline-runtime-v1", "same-cell-calibration-v1"):
        with pytest.raises(pm.ProductionManifestError, match="forbidden"):
            pm.create_sim_run_manifest_fixture(
                generation_id="forbidden-scope",
                scope=scope,
                workload_authority=workload,
                launch_evidence=(evidence,),
                replay_mode={"mode": "flat"},
                build_identity=declared,
            )


def test_stats_csv_is_closed_and_rejects_target_columns(tmp_path: Path):
    for header in (
        "total_insts,total_reported_insts,target_value,ncu_report\n54,54,1,x\n",
        "total_insts,total_reported_insts,extra\n54,54,0\n",
    ):
        bundle = tmp_path / f"stats-{sha256_bytes(header.encode())[:8]}"
        _copy_clean_bundle(bundle)
        stats_path = bundle / "trace/stats.csv"
        stats_path.write_text(header, encoding="ascii")
        _refresh_artifact_claim(bundle, "stats_csv", "trace/stats.csv")
        with pytest.raises(pm.ProductionManifestError):
            _retain_bundle(tmp_path, bundle)


def test_execution_and_control_projections_are_recursively_closed(tmp_path: Path):
    for lane, mutation in (
        ("host", lambda value: value.update({"ambient": {"host_root": "/home/alice/private/execution.json"}})),
        ("metrics", lambda value: value.update({"metrics": {"gpc__cycles_elapsed.max": 1}})),
        ("truth", lambda value: value.update({"ground_truth": {"observed_cycles": 1}})),
        ("row", lambda value: value["kernels"].append(7)),
    ):
        bundle = tmp_path / f"execution-{lane}"
        _copy_clean_bundle(bundle)
        path = bundle / "trace/extra_info/enhanced_execution_info.json"
        value = json.loads(path.read_text())
        mutation(value)
        _write_json(path, value)
        _refresh_artifact_claim(
            bundle,
            "enhanced_execution_info_json",
            "trace/extra_info/enhanced_execution_info.json",
        )
        with pytest.raises(pm.ProductionManifestError):
            _retain_bundle(tmp_path, bundle)

    for lane, mutation in (
        ("top", lambda value: value.update({"ground_truth": {"cycles": 1}})),
        ("parser", lambda value: value["parser_contract"].update({"metrics": {}})),
        ("summary", lambda value: value["summary"].update({"target_value": 1})),
        ("row", lambda value: next(iter(value["functions"].values()))[0].update({"measured": 1})),
        ("control", lambda value: next(iter(value["functions"].values()))[0]["control"].update({"reason": "x"})),
    ):
        bundle = tmp_path / f"control-{lane}"
        _copy_clean_bundle(bundle)
        path = bundle / "control/canary_sm86.control_bits.json"
        value = json.loads(path.read_text())
        mutation(value)
        _write_json(path, value)
        _refresh_artifact_claim(
            bundle,
            "control_bits_sidecar",
            "control/canary_sm86.control_bits.json",
        )
        with pytest.raises(pm.ProductionManifestError):
            _retain_bundle(tmp_path, bundle)


def test_launch_artifact_roles_reject_path_and_content_aliases(tmp_path: Path):
    path_alias = tmp_path / "path-alias"
    _copy_clean_bundle(path_alias)
    manifest_path = path_alias / "launch_evidence.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["dynamic_trace_pb"] = dict(
        manifest["artifacts"]["stats_csv"]
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(pm.ProductionManifestError, match="duplicate|partial|role"):
        _retain_bundle(tmp_path, path_alias)

    content_alias = tmp_path / "content-alias"
    _copy_clean_bundle(content_alias)
    dynamic_path = content_alias / "trace/dynamic_trace.pb"
    dynamic_path.write_bytes((content_alias / "trace/stats.csv").read_bytes())
    _refresh_artifact_claim(content_alias, "dynamic_trace_pb", "trace/dynamic_trace.pb")
    with pytest.raises(pm.ProductionManifestError, match="content claim"):
        _retain_bundle(tmp_path, content_alias)


def test_launch_evidence_cannot_be_reused_across_ordinals(tmp_path: Path):
    first_bundle = tmp_path / "first-bundle"
    second_bundle = tmp_path / "second-bundle"
    _copy_clean_bundle(first_bundle)
    _copy_clean_bundle(second_bundle)
    workload_model = _workload_model(launch_count=2)
    workload_path = _write_canonical(tmp_path / "workload.json", workload_model)
    build_path = _write_canonical(tmp_path / "build.json", _build_model())
    workload = pm.load_fixture_workload_execution_authority(workload_path)
    declared = pm.load_fixture_declared_build_identity(build_path)
    retained = []
    for index, bundle in enumerate((first_bundle, second_bundle)):
        admission_model = _admission_model(workload_model, bundle)
        admission_model.pop("model_seal_sha256")
        admission_model["existing_admission_id"] = f"reused-admission-{index}"
        admission_model["source_content_manifest_id"] = f"reused-source-{index}"
        admission = pm.load_fixture_existing_launch_admission_projection(
            canonical_json_bytes(seal_model(admission_model))
        )
        retained.append(
            pm.retain_fixture_launch_evidence(
                bundle / "launch_evidence.json",
                destination_manifest_path=(
                    f"launches/launch-{index:03d}/launch_evidence.json"
                ),
                artifact_paths=_artifact_paths(bundle),
                admission=admission,
            )
        )
    with pytest.raises(pm.ProductionManifestError, match="reused across"):
        pm.create_sim_run_manifest_fixture(
            generation_id="reused-generation",
            scope="fixture-reuse-v1",
            workload_authority=workload,
            launch_evidence=tuple(retained),
            replay_mode={"mode": "flat"},
            build_identity=declared,
        )


@pytest.mark.parametrize("lane", ["workload", "build", "admission", "producer"])
def test_post_build_mutation_cannot_influence_publication(
    tmp_path: Path, lane: str
):
    build, evidence, workload, declared, _bundle = _prepare(tmp_path)
    if lane == "workload":
        workload.document["argv"].append("changed")
        _reseal_in_place(workload.document)
    elif lane == "build":
        declared.document["source_tree_sha256"] = "0" * 64
        _reseal_in_place(declared.document)
    elif lane == "admission":
        evidence.admission.document["source_content_manifest_sha256"] = "0" * 64
        _reseal_in_place(evidence.admission.document)
    else:
        build.producer_identity["producer_sha256"] = "0" * 64
    output = tmp_path / "mutation-output"
    with pytest.raises(pm.ProductionManifestError, match="mutated|changed"):
        pm.publish_sim_run_manifest_fixture(build, output)
    assert not output.exists()


@pytest.mark.parametrize("lane", ["producer", "schema", "source", "build"])
def test_detached_loader_rederives_every_generation_claim(
    tmp_path: Path, lane: str
):
    build, *_rest = _prepare(tmp_path)
    published = pm.publish_sim_run_manifest_fixture(build, tmp_path / "published")
    forged = json.loads(published.authority_bytes)
    forged.pop("model_seal_sha256")
    if lane == "producer":
        forged["producer_identity"]["producer_sha256"] = "0" * 64
    elif lane == "schema":
        forged["schema_identities"]["sim_run_manifest_schema"]["size_bytes"] += 1
    elif lane == "source":
        forged["source_identity"]["workload_execution_authority"][
            "payload_sha256"
        ] = "0" * 64
    else:
        forged["build_identity"]["source_tree_sha256"] = "0" * 64
    forged_path = _write_canonical(
        tmp_path / f"forged-{lane}.json", seal_model(forged)
    )
    with pytest.raises(pm.ProductionManifestError):
        pm.load_sim_run_manifest_authority_fixture(
            forged_path, published.generation_path
        )


def test_projection_and_generation_omit_private_card_provenance(tmp_path: Path):
    build, *_rest = _prepare(tmp_path)
    forbidden = (
        b"data/local_labels/",
        b"/home/alice/private/",
        b"never an NCU value",
        b"target_value,ncu_report",
    )
    card_files = {
        "launches/launch-000/cards/sm_86.json",
        "launches/launch-000/cards/sm_86_static_card.json",
    }
    by_path = {item.relative_path: item.payload for item in build.generation_files}
    for path in card_files:
        assert path in by_path
        assert not any(token.lower() in by_path[path].lower() for token in forbidden)
    assert not any(
        token.lower() in payload.lower()
        for payload in by_path.values()
        for token in (b"data/local_labels/", b"/home/alice/private/")
    )


def test_mutation_between_generation_and_authority_commit_fails(
    tmp_path: Path, monkeypatch
):
    build, *_rest = _prepare(tmp_path)
    original = pm._authority_document

    def mutate_after_generation(current_build, generation):
        current_build.producer_identity["producer_sha256"] = "0" * 64
        return original(current_build, generation)

    monkeypatch.setattr(pm, "_authority_document", mutate_after_generation)
    output = tmp_path / "publication-output"
    with pytest.raises(pm.ProductionManifestError, match="mutable claim fields"):
        pm.publish_sim_run_manifest_fixture(build, output)
    assert (output / build.generation_id).is_dir()
    assert not (output / f"{build.generation_id}.authority.json").exists()


def test_manifest_replacement_after_initial_inspection_fails(
    tmp_path: Path, monkeypatch
):
    build, *_rest = _prepare(tmp_path)
    published = pm.publish_sim_run_manifest_fixture(build, tmp_path / "published")
    original_inspect = pm.inspect_immutable_directory_generation
    replaced = False

    def replace_after_inspection(path, **kwargs):
        nonlocal replaced
        generation = original_inspect(path, **kwargs)
        if not replaced and Path(path) == published.generation_path:
            replaced = True
            manifest_path = published.generation_path / pm.RUN_MANIFEST_NAME
            value = json.loads(manifest_path.read_bytes())
            value["scope"] = "fixture-canary-v2"
            replacement = published.generation_path / "replacement.json"
            os.chmod(published.generation_path, 0o755)
            replacement.write_bytes(canonical_json_bytes(value))
            os.chmod(replacement, 0o444)
            os.replace(replacement, manifest_path)
            os.chmod(published.generation_path, 0o555)
        return generation

    monkeypatch.setattr(pm, "inspect_immutable_directory_generation", replace_after_inspection)
    with pytest.raises(pm.ProductionManifestError):
        pm.load_sim_run_manifest_authority_fixture(
            published.authority_path, published.generation_path
        )


def test_detached_authority_schema_path_vocabulary_matches_trace_contract():
    schema = json.loads(pm.AUTHORITY_SCHEMA_PATH.read_bytes())
    path_validator = Draft202012Validator(schema["$defs"]["relativePath"])
    for accepted in (
        "trace/threadblocks/device_0/tb:@+!,=~.pb",
        "trace/threadblocks/device_0/tb#$%&'();[]^`{}.pb",
    ):
        path_validator.validate(accepted)
    for rejected in (
        "/trace/threadblocks/tb.pb",
        "trace//threadblocks/tb.pb",
        "trace/../threadblocks/tb.pb",
        "trace/threadblocks\\tb.pb",
        "trace/threadblocks/tb\n.pb",
    ):
        assert not path_validator.is_valid(rejected)


def test_detached_authority_schema_is_closed_and_pinned():
    payload = pm.AUTHORITY_SCHEMA_PATH.read_bytes()
    schema = json.loads(payload)
    assert sha256_bytes(payload) == pm.AUTHORITY_SCHEMA_SHA256
    assert schema["$id"] == pm.AUTHORITY_SCHEMA_ID
    assert schema["additionalProperties"] is False
    assert "labels" not in json.dumps(schema)

