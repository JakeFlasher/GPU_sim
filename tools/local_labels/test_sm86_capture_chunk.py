from __future__ import annotations

import copy
import errno
import hashlib
import json
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.local_labels import build_sm86_capture_matrix as bcm
from tools.local_labels import sm86_capture_chunk as cc
from tools.local_labels import sol_reconstitution as sr
from tools.local_labels import sol_workload_capture_chunk as sol_chunk


SHA = "a" * 64
HEAD = "b" * 40
MATRIX_SHA = "8" * 64
CERTIFYING_SOL_ENVIRONMENT = dict(cc.sol_env.PINNED_COMPILER_ENVIRONMENT)
LEGACY_CERTIFYING_SOL_ENVIRONMENT = dict(
    cc.sol_env.LEGACY_PINNED_COMPILER_ENVIRONMENT
)


@contextmanager
def _frozen_canonical_loader_context():
    """Read the frozen runnability blob without touching a held diagnostic."""
    real_load = bcm._load_json
    frozen_path = (cc.REPO_ROOT / bcm.SOL_RUNNABILITY_PATH).resolve()

    def load(path: Path) -> dict:
        if Path(path).resolve() == frozen_path:
            payload = subprocess.check_output([
                "git", "show",
                f"HEAD:{bcm.SOL_RUNNABILITY_PATH.as_posix()}",
            ], cwd=cc.REPO_ROOT)
            value = json.loads(payload)
            assert isinstance(value, dict)
            return value
        return real_load(path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bcm, "_load_json", load)
        yield


def _write_launch_observation(
    path: Path, *, report: Path, launch_index: int,
    kernel_name: str = "_Z15syntheticKernelv",
) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": launch_index,
        "requested_launch_count": 1,
        "report_sha256": cc.sha256_file(report),
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": "0",
            "Device": "0",
            "Kernel Name": kernel_name,
            "Block Size": "(128, 1, 1)",
            "Grid Size": "(1, 1, 1)",
        },
    }
    payload["observation_sha256"] = cc.canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _canonical_launch_invocation(*, selected: int = 0) -> dict:
    return {
        "capture_launch_contract": {
            "schema_version": 1,
            "label_driver_warmup_count": 1,
            "label_effective_warmup_count": 0,
            "label_launch_index": 0,
            "label_measured_reps": 1,
            "trace_launch_count": 1,
            "label_trace_launch_index": selected,
        },
        "label_app_args": [],
        "shape_delivery": "application_args",
        "logical_workload": {"value": "synthetic_shape"},
    }


def _canonical_launch(
    *, kernel_name: str = "_Z15syntheticKernelv___0",
    block: list[int] | None = None, grid: list[int] | None = None,
    device: int = 0, kernel_id: int = 1,
) -> dict:
    return {
        "device_id": device,
        "stream_id": 0,
        "kernel_id": kernel_id,
        "kernel_name": kernel_name,
        "block_dim": [128, 1, 1] if block is None else block,
        "grid_dim": [1, 1, 1] if grid is None else grid,
    }


def _canonical_observation(
    *, kernel_name: str = "_Z15syntheticKernelv",
    block: str = "(128, 1, 1)", grid: str = "(1, 1, 1)",
    device: str = "0", launch_id: str = "0",
) -> dict:
    return {
        "kernel_name_base": "mangled",
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": launch_id,
            "Device": device,
            "Kernel Name": kernel_name,
            "Block Size": block,
            "Grid Size": grid,
        },
    }


def _canonical_label(
    *, observation: dict | None = None, selectors: list[str] | None = None,
) -> dict:
    command = [str(cc.LABEL_DRIVER), "--device", "sm_86"]
    for selector in selectors or []:
        command += ["--kernel-name", selector]
    command += ["--", "synthetic-target-arg"]
    return {
        "command": command,
        "ncu_launch_observation_document": (
            _canonical_observation() if observation is None else observation
        ),
    }


@pytest.mark.parametrize(
    "block,grid,device",
    [
        ("(128, 1, 1)", "(1, 1, 1)", "0"),
        (" 128, 1, 1 ", "1,1,1", "NVIDIA GeForce RTX 3090 Ti (0)"),
    ],
)
def test_canonical_launch_identity_join_accepts_exact_independent_tuple(
    block, grid, device,
):
    observation = _canonical_observation(block=block, grid=grid, device=device)
    binding = cc.require_canonical_label_launch_identity(
        label=_canonical_label(observation=observation),
        trace={"launches": [_canonical_launch()]},
        invocation=_canonical_launch_invocation(),
    )
    assert binding == {
        "launch_key": (
            "_Z15syntheticKernelv", (128, 1, 1), (1, 1, 1), 0,
        ),
        "kernel_selector": None,
    }


@pytest.mark.parametrize("field", ["name", "block", "grid", "device"])
def test_canonical_launch_identity_join_refuses_each_tuple_mismatch(field):
    observation = _canonical_observation()
    replacements = {
        "name": ("Kernel Name", "_Z9differentv"),
        "block": ("Block Size", "(64, 1, 1)"),
        "grid": ("Grid Size", "(2, 1, 1)"),
        "device": ("Device", "1"),
    }
    key, value = replacements[field]
    observation["identity"][key] = value
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.require_canonical_label_launch_identity(
            label=_canonical_label(observation=observation),
            trace={"launches": [_canonical_launch()]},
            invocation=_canonical_launch_invocation(),
        )
    assert caught.value.code == "label_launch_identity_join_mismatch"


@pytest.mark.parametrize("mutation", ["missing", "blank", "duplicate"])
def test_canonical_launch_identity_join_requires_unambiguous_ncu_id(mutation):
    observation = _canonical_observation()
    if mutation == "missing":
        observation["identity"].pop("ID")
        observation["identity_columns"].remove("ID")
    elif mutation == "blank":
        observation["identity"]["ID"] = ""
    else:
        observation["identity_columns"].append("ID")
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.require_canonical_label_launch_identity(
            label=_canonical_label(observation=observation),
            trace={"launches": [_canonical_launch()]},
            invocation=_canonical_launch_invocation(),
        )
    assert caught.value.code == "label_launch_identity_join_mismatch"


@pytest.mark.parametrize(
    "kernel_name", [
        "_Z15syntheticKernelv", "___0", "_Z15syntheticKernelv___0___0",
        "regex:.*___0",
    ],
)
def test_canonical_launch_identity_join_refuses_malformed_trace_suffix(kernel_name):
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.require_canonical_label_launch_identity(
            label=_canonical_label(),
            trace={"launches": [_canonical_launch(kernel_name=kernel_name)]},
            invocation=_canonical_launch_invocation(),
        )
    assert caught.value.code == "label_launch_identity_join_mismatch"


def test_canonical_launch_identity_join_refuses_out_of_range_selected_row():
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.require_canonical_label_launch_identity(
            label=_canonical_label(),
            trace={"launches": [_canonical_launch()]},
            invocation=_canonical_launch_invocation(selected=1),
        )
    assert caught.value.code == "label_launch_identity_join_mismatch"


def test_equal_ncu_and_nvbit_ordinals_do_not_override_identity_mismatch():
    observation = _canonical_observation(launch_id="0", kernel_name="_Z9differentv")
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.require_canonical_label_launch_identity(
            label=_canonical_label(observation=observation),
            trace={"launches": [_canonical_launch(kernel_id=0)]},
            invocation=_canonical_launch_invocation(),
        )
    assert caught.value.code == "label_launch_identity_join_mismatch"


def _warm_cold_invocation() -> dict:
    sequence = [
        {
            "device_id": 0, "stream_id": 0, "kernel_id": 1,
            "kernel_name": "warm_cold_pair_kernel___0",
            "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
            "role": "cold",
        },
        {
            "device_id": 0, "stream_id": 0, "kernel_id": 2,
            "kernel_name": "warm_cold_pair_eviction_kernel___0",
            "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
            "role": "evict",
        },
        {
            "device_id": 0, "stream_id": 0, "kernel_id": 3,
            "kernel_name": "warm_cold_pair_kernel___0",
            "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
            "role": "warm",
        },
    ]
    invocation = _canonical_launch_invocation()
    invocation["capture_launch_contract"].update({
        "trace_launch_count": 3,
        "trace_launch_sequence": sequence,
    })
    return invocation


def test_warm_cold_contract_derives_and_joins_exact_mangled_selector():
    invocation = _warm_cold_invocation()
    launches = invocation["capture_launch_contract"]["trace_launch_sequence"]
    selector = cc._canonical_label_kernel_selector(invocation)
    assert selector == "warm_cold_pair_kernel"
    binding = cc.require_canonical_label_launch_identity(
        label=_canonical_label(
            observation=_canonical_observation(
                kernel_name=selector,
                block="(32, 1, 1)", grid="(32, 1, 1)",
            ),
            selectors=[selector],
        ),
        trace={"launches": launches},
        invocation=invocation,
    )
    assert binding["kernel_selector"] == selector


@pytest.mark.parametrize(
    "selectors",
    [[], ["warm_cold_pair_eviction_kernel"],
     ["warm_cold_pair_kernel", "warm_cold_pair_kernel"]],
)
def test_warm_cold_join_refuses_missing_wrong_or_duplicate_selector(selectors):
    invocation = _warm_cold_invocation()
    launches = invocation["capture_launch_contract"]["trace_launch_sequence"]
    observation = _canonical_observation(
        kernel_name="warm_cold_pair_kernel",
        block="(32, 1, 1)", grid="(32, 1, 1)",
    )
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.require_canonical_label_launch_identity(
            label=_canonical_label(observation=observation, selectors=selectors),
            trace={"launches": launches},
            invocation=invocation,
        )
    assert caught.value.code == "label_launch_identity_join_mismatch"


def test_single_launch_contract_derives_no_kernel_selector():
    assert cc._canonical_label_kernel_selector(_canonical_launch_invocation()) is None


def test_label_driver_forwards_exact_selector_with_mangled_name_base():
    source = cc.LABEL_DRIVER.read_text()
    assert (
        'NCU_COLLECT_ARGS+=(--kernel-name-base mangled --kernel-name "$KERNEL_NAME")'
        in source
    )
    assert 'if [[ "$KERNEL_NAME" == regex:* ]]' in source


def test_label_driver_refuses_regex_selector_before_capture(tmp_path):
    completed = subprocess.run(
        [
            str(cc.LABEL_DRIVER),
            "--device", "sm_86",
            "--slug", "synthetic",
            "--shape-point", "synthetic_shape",
            "--repeat", "0",
            "--launch", "0",
            "--binary", "/bin/true",
            "--kernel-name", "regex:.*",
            "--out-dir", str(tmp_path),
        ],
        env={**cc.os.environ, "LABEL_DRIVER_SKIP_ARCH_CHECK": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "requires an exact name, not regex: syntax" in completed.stderr
    assert not any(tmp_path.iterdir())


def test_capture_cell_refuses_identity_mismatch_before_control(tmp_path, monkeypatch):
    binary = tmp_path / "synthetic-binary"
    binary.write_bytes(b"synthetic executable")
    invocation = _canonical_launch_invocation()
    invocation["executable_sha256"] = cc.sha256_file(binary)
    cell = {
        "identity": {"roster_source": "portable_lcd", "slug": "synthetic"},
        "control_bits": {"policy": "direct_executable"},
    }
    budget = SimpleNamespace(max_wall_time_s=10)
    cell_dir = tmp_path / "chunk" / "cells" / "portable_lcd__synthetic"
    cell_dir.mkdir(parents=True)
    monkeypatch.setattr(
        cc, "validate_capture_cell",
        lambda *_args, **_kwargs: (binary, invocation, budget, None, None),
    )
    monkeypatch.setattr(cc, "live_capture_guard", lambda _pins: {"status": "ok"})

    def fake_label(**kwargs):
        assert kwargs["kernel_name_selector"] is None
        return _canonical_label(
            observation=_canonical_observation(kernel_name="_Z9differentv")
        )

    monkeypatch.setattr(cc, "capture_label", fake_label)
    monkeypatch.setattr(
        cc, "capture_trace",
        lambda **_kwargs: {"launches": [_canonical_launch()]},
    )
    control_called = False

    def unexpected_control(**_kwargs):
        nonlocal control_called
        control_called = True
        pytest.fail("control capture ran after an identity mismatch")

    monkeypatch.setattr(cc, "capture_control", unexpected_control)
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.capture_cell(
            cell=cell,
            cell_dir=cell_dir,
            matrix={},
            pins={},
            cell_budgets=object(),
            expected_tracer_sha256="9" * 64,
            capture_provenance={
                "capture_started_at_utc": "2026-07-10T12:00:00Z",
                "admission_run_id": "8" * 64,
                "supersedes": None,
                "prior_preparation_failures": [],
                "prior_capture_defect_attempts": [],
                "source_commit": "d" * 40,
            },
            expected_ncu_identity=_expected_ncu_identity(),
            certified_sol_env=None,
            sol_environment_probe=None,
            module_store=tmp_path / "module-store",
        )
    assert caught.value.code == "label_launch_identity_join_mismatch"
    assert control_called is False


@pytest.mark.parametrize(
    "identity,expected_code",
    [
        (
            {"roster_source": "sol_execbench", "slug": "synthetic"},
            "sol_supplemental_launch_set_v3_live_canary_required",
        ),
        (
            {"roster_source": "portable_lcd", "slug": "synthetic"},
            "sol_workload_row_binding_invalid",
        ),
        (None, "sol_workload_row_binding_invalid"),
    ],
)
def test_supplemental_cell_gate_refuses_before_validation_or_capture(
    tmp_path, monkeypatch, identity, expected_code,
):
    cell_dir = tmp_path / "chunk" / "cells" / "must-not-exist"

    def unexpected(*_args, **_kwargs):
        pytest.fail("supplemental cell crossed the live-canary gate")

    for name in (
        "validate_capture_cell",
        "_write_json_exclusive",
        "live_capture_guard",
        "capture_label",
        "capture_trace",
        "_run_logged",
    ):
        monkeypatch.setattr(cc, name, unexpected)
    monkeypatch.setattr(cc.subprocess, "run", unexpected)

    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.capture_cell(
            cell={"identity": identity},
            cell_dir=cell_dir,
            matrix={},
            pins={},
            cell_budgets=object(),
            expected_tracer_sha256="9" * 64,
            capture_provenance={
                "capture_started_at_utc": "2026-07-10T12:00:00Z",
                "admission_run_id": "8" * 64,
                "supersedes": None,
                "prior_preparation_failures": [],
                "prior_capture_defect_attempts": [],
                "source_commit": "d" * 40,
            },
            expected_ncu_identity=_expected_ncu_identity(),
            certified_sol_env=None,
            sol_environment_probe=None,
            module_store=tmp_path / "module-store",
            workload_row_binding={"schema_version": 1},
        )
    assert caught.value.code == expected_code
    assert not cell_dir.exists()


def test_vendor_runtime_control_slug_dispatch_is_closed():
    expected = frozenset({
        "vendor_gemm_cublas",
        "vendor_sgemm_cublas",
        "vendor_sgemm_strided_batched_cublas",
    })
    assert cc._VENDOR_RUNTIME_CONTROL_SLUGS == expected
    assert frozenset(bcm.VENDOR_RUNTIME_CONTROL_CONTRACTS) == expected


def _cell(source: str = "portable_lcd", slug: str = "silu_fwd_bf16_sm80") -> dict:
    return {
        "identity": {"roster_source": source, "slug": slug},
        "authority_cell_sha256": SHA,
        "disposition": "runnable",
        "capture_readiness": {"status": "ready", "blocking_reasons": []},
    }


def _matrix() -> dict:
    members = [{"roster_source": "portable_lcd", "slug": "silu_fwd_bf16_sm80"}]
    return {
        "source_commit": HEAD,
        "cells": [_cell()],
        "chunks": [{
            "chunk_id": "portable_lcd-000-bootstrap",
            "membership_sha256": cc.canonical_sha256(members),
            "cells": members,
        }],
    }


def test_select_chunk_binds_exact_nonempty_membership():
    chunk, cells = cc.select_chunk(_matrix(), "portable_lcd-000-bootstrap")
    assert chunk["membership_sha256"]
    assert [row["identity"]["slug"] for row in cells] == ["silu_fwd_bf16_sm80"]


def test_select_chunk_refuses_empty_membership():
    matrix = _matrix()
    matrix["chunks"][0]["cells"] = []
    matrix["chunks"][0]["membership_sha256"] = cc.canonical_sha256([])
    with pytest.raises(cc.ChunkRefusal, match="chunk_empty"):
        cc.select_chunk(matrix, "portable_lcd-000-bootstrap")


def test_select_chunk_refuses_duplicate_member():
    matrix = _matrix()
    members = matrix["chunks"][0]["cells"] * 2
    matrix["chunks"][0]["cells"] = members
    matrix["chunks"][0]["membership_sha256"] = cc.canonical_sha256(members)
    with pytest.raises(cc.ChunkRefusal, match="chunk_member_duplicate"):
        cc.select_chunk(matrix, "portable_lcd-000-bootstrap")


def test_select_chunk_refuses_hash_tamper():
    matrix = _matrix()
    matrix["chunks"][0]["membership_sha256"] = SHA
    with pytest.raises(cc.ChunkRefusal, match="chunk_membership_hash_mismatch"):
        cc.select_chunk(matrix, "portable_lcd-000-bootstrap")


def test_canonical_hash_matches_utf8_generator_contract():
    assert cc.canonical_sha256({"detail": "correctness — failed"}) == (
        "712cc55941ab237f90f7d0fb19aa845a22cf1281853a7a46f665f39d9e0a9cd2"
    )


def test_sol_certifying_environment_dispatches_tracked_legacy_and_live_exactly():
    matrix = json.loads((
        cc.REPO_ROOT / "data/local_labels/campaign"
        / "sm86_reference_capture_matrix.json"
    ).read_text())
    canonical = next(
        cell["invocation"]
        for cell in matrix["cells"]
        if cell.get("disposition") == "runnable"
        and (cell.get("identity") or {}).get("roster_source") == "sol_execbench"
    )
    contract, digest = cc._sol_certifying_environment(canonical)
    assert contract == LEGACY_CERTIFYING_SOL_ENVIRONMENT
    assert digest == (
        "d959a1a89d2f3aa3e7c351ba59788c016404bbc77e9da428123188ada775af74"
    )

    live = {
        "adapter_contract": {
            "environment": copy.deepcopy(
                sol_chunk.workload_matrix.EXACT_SOL_ENVIRONMENT
            )
        }
    }
    live_contract, live_digest = cc._sol_certifying_environment(live)
    assert live_contract == CERTIFYING_SOL_ENVIRONMENT
    assert live_digest == cc.sol_env.environment_contract_sha256(
        CERTIFYING_SOL_ENVIRONMENT
    )

    malformed = copy.deepcopy(canonical)
    malformed["adapter_contract"]["environment"]["CC"] = "gcc-14"
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc._sol_certifying_environment(malformed)
    assert caught.value.code == "SOL_ENV_MATRIX_CONTRACT_INVALID"


def test_select_chunk_refuses_excluded_or_unready():
    matrix = _matrix()
    matrix["cells"][0]["disposition"] = "excluded"
    with pytest.raises(cc.ChunkRefusal, match="chunk_member_not_runnable"):
        cc.select_chunk(matrix, "portable_lcd-000-bootstrap")
    matrix = _matrix()
    matrix["cells"][0]["capture_readiness"] = {
        "status": "blocked", "blocking_reasons": ["budget_missing"]
    }
    with pytest.raises(cc.ChunkRefusal, match="chunk_member_not_ready"):
        cc.select_chunk(matrix, "portable_lcd-000-bootstrap")


def _preflight(root: Path) -> dict:
    checks = {name: {"status": "ok"} for name in cc.preflight._BLOCKING_CHECKS}
    version = "Version 2026.1.0.0 (public-release)"
    launcher = "/opt/nvidia/nsight-compute/2026.1.0/ncu"
    checks["ncu"].update({
        "expected": "2026.1.0",
        "observed": version,
        "execution_mode": cc.preflight.NCU_EXECUTION_MODE,
        "host_machine": cc.preflight.NCU_HOST_MACHINE,
        "launcher_resolved_path": launcher,
        "launcher_sha256": "4" * 64,
        "launcher_target_relpath": cc.preflight.NCU_LAUNCHER_TARGET_RELPATH,
        "resolved_path": str(
            Path(launcher).parent / cc.preflight.NCU_LAUNCHER_TARGET_RELPATH
        ),
        "executable_sha256": "5" * 64,
        "version_line": version,
    })
    checks["trace_budget"].update({
        "output_root": str(root), "budget_sha256": "9" * 64,
    })
    checks["injection_runtime"].update({
        "admitted_insts": 4800, "expected_insts": 4800,
        "validator_ok": True, "threadblocks_pb_count": 4,
        "root_trace_ok": True, "root_trace_pb_bytes": 100,
        "root_trace_contract_id": "sm86_addr_carry_v1",
        "root_trace_memory_payload_classes": [1, 2, 3, 6, 8],
        "root_trace_launch_count": 1,
        "full_coverage_ok": True, "grid_threadblock_count": 4,
        "decoded_threadblock_count": 4, "decoded_record_count": 1632,
        "zero_record_threadblock_count": 0,
        "sampling_sidecar_present": False, "stats_row_count": 1,
        "tracer_so_sha256": "c" * 64, "smoke_binary_sha256": "d" * 64,
        "dynamic_trace_pb_sha256": "e" * 64, "stats_csv_sha256": "f" * 64,
    })
    checks["sol_environment"].update({
        "typed_error": None,
        "ambient": {},
        "environment_contract": {
            "CUTLASS_DIR": "third_party/cutlass",
            "PATH_prepend": "tools/sol_execbench/.venv/bin",
            **CERTIFYING_SOL_ENVIRONMENT,
        },
        "environment_contract_sha256": "1" * 64,
        "constructed_environment_delta": dict(CERTIFYING_SOL_ENVIRONMENT),
        "matrix_pin": {
            "path": (
                "data/local_labels/campaign/"
                "sol_native_env_fingerprint_sm86.json"
            ),
            "sha256": "2" * 64,
            "native_env_hash": "3" * 64,
        },
        "matrix_sha256": MATRIX_SHA,
        "observed_fingerprint": {
            "components": {},
            "hash": "3" * 64,
            "missing": [],
            "degraded": False,
        },
        "compiler_identity": {
            "requested": "g++-14",
            "resolved_realpath": "/usr/bin/g++-14",
            "executable_sha256": "4" * 64,
            "version_line": "g++-14 synthetic test compiler",
        },
        "nvcc_release": {"major": 13, "minor": 2},
        "nvcc_version_output": "Cuda compilation tools, release 13.2",
    })
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 3,
        "captured_at_utc": generated_at,
        "generated_at_utc": generated_at,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "protocol_version": "v2.1",
        "capture_admitted": True,
        "working_tree_clean": True,
        "git_head": HEAD,
        "pins_sha256": SHA,
        "capture_matrix_sha256": MATRIX_SHA,
        "blockers": [],
        "checks": checks,
    }


def _expected_ncu_identity():
    return cc.ncu_executor.ExpectedNcuIdentity(
        resolved_path=(
            "/opt/nvidia/nsight-compute/2026.1.0/"
            "target/linux-desktop-glibc_2_11_3-x64/ncu"
        ),
        executable_sha256="5" * 64,
        version_line="Version 2026.1.0.0 (public-release)",
    )


def test_preflight_returns_exact_validated_ncu_identity(tmp_path):
    observed = cc.validate_preflight_manifest(
        _preflight(tmp_path), expected_head=HEAD, expected_pins_sha256=SHA,
        expected_output_root=tmp_path,
        expected_ncu_version_prefix="2026.1.0",
    )
    assert observed == _expected_ncu_identity()
    assert type(observed) is cc.ncu_executor.ExpectedNcuIdentity


def _historical_v2_preflight(root: Path) -> dict:
    manifest = _preflight(root)
    manifest["schema_version"] = 2
    ncu = manifest["checks"]["ncu"]
    ncu["resolved_path"] = ncu["launcher_resolved_path"]
    ncu["executable_sha256"] = ncu["launcher_sha256"]
    for field in (
        "execution_mode", "host_machine", "launcher_resolved_path",
        "launcher_sha256", "launcher_target_relpath",
    ):
        del ncu[field]
    return manifest


def test_historical_v2_preflight_is_finalization_only(tmp_path):
    manifest = _historical_v2_preflight(tmp_path)
    assert cc.validate_preflight_manifest(
        manifest, expected_head=HEAD, expected_pins_sha256=SHA,
        expected_output_root=tmp_path,
        expected_ncu_version_prefix="2026.1.0", require_fresh=False,
    ) is None
    with pytest.raises(
        cc.ChunkRefusal, match="historical-only and cannot authorize capture",
    ):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0", require_fresh=True,
        )


def test_preflight_refuses_inner_path_outside_launcher_target(tmp_path):
    manifest = _preflight(tmp_path)
    manifest["checks"]["ncu"]["resolved_path"] = "/opt/other/ncu"
    with pytest.raises(
        cc.ChunkRefusal, match="inner path does not match launcher target",
    ):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0", require_fresh=False,
        )


def test_preflight_structure_only_validation_still_rejects_ncu_authority(tmp_path):
    manifest = _preflight(tmp_path)
    manifest["checks"]["ncu"]["resolved_path"] = "//opt/ncu"
    with pytest.raises(cc.ChunkRefusal, match="preflight_invalid"):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0", require_fresh=False,
        )


def test_preflight_refuses_stale_timestamp(tmp_path):
    manifest = _preflight(tmp_path)
    manifest["captured_at_utc"] = (
        datetime.now(timezone.utc) - timedelta(minutes=16)
    ).isoformat()
    manifest["generated_at_utc"] = manifest["captured_at_utc"]
    with pytest.raises(cc.ChunkRefusal, match="PREFLIGHT_EXPIRED"):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0",
        )
    assert cc.validate_preflight_manifest(
        manifest, expected_head=HEAD, expected_pins_sha256=SHA,
        expected_output_root=tmp_path,
        expected_ncu_version_prefix="2026.1.0", require_fresh=False,
    ) is None


@pytest.mark.parametrize("age_seconds", [899.999, 900.0])
def test_preflight_freshness_inclusive_boundary_admits(tmp_path, age_seconds):
    admitted = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
    manifest = _preflight(tmp_path)
    generated = (admitted - timedelta(seconds=age_seconds)).isoformat()
    manifest["captured_at_utc"] = generated
    manifest["generated_at_utc"] = generated
    cc.validate_preflight_manifest(
        manifest, expected_head=HEAD, expected_pins_sha256=SHA,
        expected_output_root=tmp_path,
        expected_ncu_version_prefix="2026.1.0",
        admitted_at_utc=admitted,
    )


def test_preflight_900_001_seconds_old_refuses(tmp_path):
    admitted = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
    manifest = _preflight(tmp_path)
    generated = (admitted - timedelta(seconds=900.001)).isoformat()
    manifest["captured_at_utc"] = generated
    manifest["generated_at_utc"] = generated
    with pytest.raises(cc.ChunkRefusal, match="PREFLIGHT_EXPIRED"):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0",
            admitted_at_utc=admitted,
        )


def test_preflight_more_than_two_seconds_in_future_refuses_clock_skew(tmp_path):
    admitted = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
    manifest = _preflight(tmp_path)
    generated = (admitted + timedelta(seconds=2.001)).isoformat()
    manifest["captured_at_utc"] = generated
    manifest["generated_at_utc"] = generated
    with pytest.raises(cc.ChunkRefusal, match="PREFLIGHT_CLOCK_SKEW"):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0",
            admitted_at_utc=admitted,
        )


def test_preflight_requires_exact_head_pins_and_output_root(tmp_path):
    cc.validate_preflight_manifest(
        _preflight(tmp_path), expected_head=HEAD, expected_pins_sha256=SHA,
        expected_output_root=tmp_path,
        expected_ncu_version_prefix="2026.1.0",
    )
    bad = _preflight(tmp_path)
    bad["git_head"] = "c" * 40
    with pytest.raises(cc.ChunkRefusal, match="preflight_invalid"):
        cc.validate_preflight_manifest(
            bad, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy_schema", "missing_hash", "expected", "observed", "relative_path",
        "hash", "version", "unexpected",
    ],
)
def test_preflight_refuses_invalid_ncu_executable_authority(tmp_path, mutation):
    manifest = _preflight(tmp_path)
    ncu = manifest["checks"]["ncu"]
    if mutation == "legacy_schema":
        manifest["schema_version"] = 1
    elif mutation == "missing_hash":
        ncu.pop("executable_sha256")
    elif mutation == "expected":
        ncu["expected"] = "2026.2.0"
    elif mutation == "observed":
        ncu["observed"] = "Version 2026.1.0.1 (different)"
    elif mutation == "relative_path":
        ncu["resolved_path"] = "relative/ncu"
    elif mutation == "hash":
        ncu["executable_sha256"] = "not-a-hash"
    elif mutation == "version":
        ncu["observed"] = "Version 2026.10.0"
        ncu["version_line"] = "Version 2026.10.0"
    else:
        ncu["self_attested"] = True
    with pytest.raises(cc.ChunkRefusal, match="preflight_invalid"):
        cc.validate_preflight_manifest(
            manifest,
            expected_head=HEAD,
            expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0",
        )


def test_preflight_refuses_invalid_independent_ncu_version_pin(tmp_path):
    with pytest.raises(
        cc.ChunkRefusal, match="expected NCU version authority is invalid",
    ):
        cc.validate_preflight_manifest(
            _preflight(tmp_path),
            expected_head=HEAD,
            expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.beta",
        )


@pytest.mark.parametrize("field,value", [
    ("capture_admitted", False),
    ("working_tree_clean", False),
    ("blockers", ["clock_state"]),
])
def test_preflight_refuses_self_asserted_blocked_state(tmp_path, field, value):
    manifest = _preflight(tmp_path)
    manifest[field] = value
    with pytest.raises(cc.ChunkRefusal, match="preflight_invalid"):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0",
        )


def test_preflight_refuses_one_non_ok_probe(tmp_path):
    manifest = _preflight(tmp_path)
    manifest["checks"]["injection_runtime"]["status"] = "failed"
    with pytest.raises(cc.ChunkRefusal, match="checks.injection_runtime"):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0",
        )


def test_preflight_rederives_and_rejects_contradictory_coverage(tmp_path):
    manifest = _preflight(tmp_path)
    manifest["checks"]["injection_runtime"].update({
        "full_coverage_ok": True,
        "decoded_threadblock_count": 3,
        "threadblocks_pb_count": 3,
    })
    with pytest.raises(cc.ChunkRefusal, match="contradictory"):
        cc.validate_preflight_manifest(
            manifest, expected_head=HEAD, expected_pins_sha256=SHA,
            expected_output_root=tmp_path,
            expected_ncu_version_prefix="2026.1.0",
        )


def test_artifact_row_hashes_relative_regular_file(tmp_path):
    root = tmp_path / "root"
    path = root / "nested" / "a.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"payload")
    row = cc.artifact_row(path, root=root, artifact_kind="trace_root")
    assert row == {
        "relative_path": "nested/a.bin",
        "sha256": hashlib.sha256(b"payload").hexdigest(),
        "size_bytes": 7,
        "artifact_kind": "trace_root",
    }


def test_artifact_row_refuses_path_escape_and_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("x")
    with pytest.raises(cc.ChunkRefusal, match="artifact_path_escape"):
        cc.artifact_row(outside, root=root, artifact_kind="trace_root")
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(cc.ChunkRefusal, match="artifact_not_regular"):
        cc.artifact_row(link, root=root, artifact_kind="trace_root")


def test_inventory_hashes_complete_trace_envelope_and_logs(tmp_path):
    chunk = tmp_path / "chunk"
    cell = chunk / "cells" / "portable__silu"
    paths = {
        "sidecar": cell / "labels" / "label_0_0.json",
        "run_manifest": cell / "labels" / "label_run_manifest_0_0.json",
        "ncu_report": cell / "labels" / "run_0_0.ncu-rep",
        "ncu_launch_observation": (
            cell / "labels" / "run_0_0.launch_observation.json"
        ),
        "trace_manifest": cell / "trace_capture_manifest.json",
        "root": cell / "trace" / "dynamic_trace.pb",
        "stats": cell / "trace" / "stats.csv",
        "tb": cell / "trace" / "threadblocks" / "kernel_0" / "0.pb",
        "alloc": cell / "trace" / "extra_info" / "allocations.json",
        "control": cell / "control_bits.json",
        "control_manifest": cell / "control_capture_manifest.json",
        "result": cell / "cell_capture_result.json",
        "log": cell / "logs" / "trace.stderr.log",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    rows = cc.inventory_cell_artifacts(
        cell_dir=cell, chunk_root=chunk,
        label={
            k: paths[k] for k in (
                "sidecar", "run_manifest", "ncu_report",
                "ncu_launch_observation",
            )
        },
        trace={"manifest": paths["trace_manifest"], "root_pb": paths["root"],
               "stats_csv": paths["stats"]},
        control={"control_bits": paths["control"],
                 "manifest": paths["control_manifest"]},
        result_path=paths["result"],
        disposition="admitted", typed_reason=None,
    )
    assert len(rows) == len(paths)
    kinds = {row["artifact_kind"] for row in rows}
    assert {"trace_threadblock", "trace_auxiliary", "execution_log"} <= kinds
    from tools.local_labels import campaign_done_manifest as done
    normalized = done._normalize_results([{
        "roster_source": "portable_lcd",
        "slug": "silu_fwd_bf16_sm80",
        "authority_cell_sha256": SHA,
        "disposition": "admitted",
        "capture_preflight_sha256": "1" * 64,
        "capture_started_at_utc": "2026-07-10T12:00:01Z",
        "admission_run_id": "2" * 64,
        "supersedes": None,
        "prior_preparation_failures": [],
        "prior_capture_defect_attempts": [],
        "definition_equivalence": None,
        "definition_equivalence_attestation_sha256": None,
        "artifacts": rows,
    }])
    assert normalized[0]["disposition"] == "admitted"


def test_trace_tree_bytes_refuses_symlink(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    target = tmp_path / "outside"
    target.write_text("x")
    (trace / "link").symlink_to(target)
    with pytest.raises(cc.ChunkRefusal, match="trace_symlink_refused"):
        cc._trace_tree_bytes(trace)


def test_validate_direct_cell_rehashes_binary_and_requires_explicit_budget(tmp_path, monkeypatch):
    binary = tmp_path / "run"
    binary.write_text("binary")
    binary.chmod(0o755)
    digest = cc.sha256_file(binary)
    cell = _cell()
    cell.update({
        "trace_budget": {
            "max_trace_bytes": 100, "max_wall_time_s": 10,
            "sampling_policy": "off",
        },
        "invocation": {
            "adapter_kind": "direct_executable",
            "executable_path": str(binary), "executable_sha256": digest,
            "label_app_args": ["--timing_shape"],
            "trace_app_args": ["--timing_shape", "--warmup", "0", "--reps", "1"],
            "shape_delivery": "application_args",
            "logical_workload": {"kind": "shape_point", "value": "timing_shape"},
            "capture_launch_contract": {
                "schema_version": 1, "label_driver_warmup_count": 1,
                "label_effective_warmup_count": 1, "label_launch_index": 1,
                "label_measured_reps": 1, "trace_launch_count": 1,
                "label_trace_launch_index": 0,
            },
        },
        "control_bits": {"policy": "direct_executable", "input_sha256": digest},
    })
    budget = SimpleNamespace(
        max_trace_bytes=100, max_wall_time_s=10, sampling_policy="off"
    )
    monkeypatch.setattr(cc, "require_cell_budget", lambda *_: budget)
    got, invocation, got_budget = cc.validate_direct_cell(cell, cell_budgets=object())
    assert got == binary and invocation["trace_app_args"][-1] == "1"
    assert got_budget is budget
    binary.write_text("tampered")
    with pytest.raises(cc.ChunkRefusal, match="input_hash_mismatch"):
        cc.validate_direct_cell(cell, cell_budgets=object())


def test_capture_launch_contract_rejects_duplicate_or_malformed_warmup():
    base = {
        "shape_delivery": "application_args",
        "logical_workload": {"kind": "shape_point", "value": "trace_00"},
        "capture_launch_contract": {
            "schema_version": 1, "label_driver_warmup_count": 1,
            "label_effective_warmup_count": 1, "label_launch_index": 1,
            "label_measured_reps": 1, "trace_launch_count": 1,
            "label_trace_launch_index": 0,
        },
    }
    duplicate = copy.deepcopy(base)
    duplicate["label_app_args"] = ["--warmup", "1", "--warmup=1"]
    with pytest.raises(cc.ChunkRefusal, match="invocation_flag_invalid"):
        cc._require_capture_launch_contract(duplicate)
    malformed = copy.deepcopy(base)
    malformed["label_app_args"] = ["--warmup", "one"]
    with pytest.raises(cc.ChunkRefusal, match="invocation_flag_invalid"):
        cc._require_capture_launch_contract(malformed)


def test_warm_cold_trace_launch_contract_requires_exact_sequence():
    matrix = json.loads(cc.DEFAULT_MATRIX.read_text())
    invocation = next(
        cell["invocation"] for cell in matrix["cells"]
        if cell["identity"] == {
            "roster_source": "simbench_bw", "slug": "warm_cold_pair_relaunch",
        }
    )
    sequence = invocation["capture_launch_contract"]["trace_launch_sequence"]
    root = {
        "launch_count": 3,
        "launches": [
            {key: row[key] for key in (
                "device_id", "stream_id", "kernel_id", "kernel_name",
                "grid_dim", "block_dim",
            )}
            for row in sequence
        ],
    }
    cc._validate_trace_launch_contract(invocation, root)
    wrong_order = copy.deepcopy(root)
    wrong_order["launches"].reverse()
    with pytest.raises(cc.ChunkRefusal, match="trace_launch_sequence_invalid"):
        cc._validate_trace_launch_contract(invocation, wrong_order)
    wrong_count = copy.deepcopy(root)
    wrong_count["launch_count"] = 2
    with pytest.raises(cc.ChunkRefusal, match="trace_launch_cardinality_invalid"):
        cc._validate_trace_launch_contract(invocation, wrong_count)


def _generated_sol_cell(
    slug_prefix: str = "problem_001_",
) -> tuple[dict, dict]:
    """Build one strict current-format SOL cell without live sweep artifacts.

    Unit tests must not become contingent on a partially completed 201-cell GPU
    run.  The fixture rederives the invocation from immutable authority/input
    bytes and a closed synthetic *passing* evidence row; production loaders are
    never patched or weakened.
    """
    matrix = json.loads(cc.DEFAULT_MATRIX.read_text())
    authority = json.loads(cc.DEFAULT_AUTHORITY.read_text())
    authority_cell = copy.deepcopy(next(
        row for row in authority["cells"]
        if row["resolution_status"] == "runnable"
        and row["roster_source"] == "sol_execbench"
        and row["slug"].startswith(slug_prefix)
    ))
    identity = {
        "roster_source": "sol_execbench", "slug": authority_cell["slug"],
    }
    category = authority_cell["category"]
    fingerprint_path = cc.REPO_ROOT / bcm.SOL_NATIVE_ENV_FINGERPRINT_PATH
    fingerprint_payload = fingerprint_path.read_bytes()
    fingerprint = json.loads(fingerprint_payload)
    native_env_hash = fingerprint["native_env_hash"]
    authority_cell["native_env_hash"] = native_env_hash

    solution = cc.REPO_ROOT / authority_cell["binary"]
    solution_document = json.loads(solution.read_text())
    benchmark = (
        cc.REPO_ROOT / bcm.SOL_BENCHMARK_ROOT / bcm.SOL_SUBSET_MAP[category]
        / solution_document["definition"]
    )
    expected_workloads = bcm._count_workloads(benchmark / "workload.jsonl")
    evidence_path = (
        cc.REPO_ROOT / "tools/local_labels/testdata/sol_runnability_unit_fixture.json"
    )
    evidence_document = json.loads(evidence_path.read_text())
    corpus_probe = {
        "flashinfer_corpus_verified": True,
        "_flashinfer_corpus_root": sr.FLASHINFER_CORPUS_ROOT,
        "data_annex_manifest_sha256": (
            sr.DATA_ANNEX_MANIFEST_SHA256
        ),
        "flashinfer_corpus_manifest_sha256": (
            sr.FLASHINFER_CORPUS_MANIFEST_SHA256
        ),
        "flashinfer_corpus_file_count": (
            sr.FLASHINFER_CORPUS_FILE_COUNT
        ),
        "flashinfer_corpus_blob_count": (
            sr.FLASHINFER_CORPUS_BLOB_COUNT
        ),
    }
    run_row = copy.deepcopy(next(
        row for row in evidence_document["cells"]
        if row["slug"] == identity["slug"]
    ))
    assert run_row == {
        "slug": identity["slug"],
        "category": category,
        "expected_workloads": expected_workloads,
        "status": "runnable",
        "returncode": 0,
        "detail": f"{expected_workloads}/{expected_workloads} workloads passed",
    }
    run_contract = {
        "source_sha256": bcm.sha256_file(evidence_path),
        "native_env_hash": native_env_hash,
        "rows_by_slug": {identity["slug"]: run_row},
        "flashinfer_corpus_by_category": {
            category: sr.flashinfer_capture_contract(
                corpus_probe, category=category,
            ),
        },
    }
    invocation, control = bcm._sol_invocation(
        authority_cell, cc.REPO_ROOT,
        shrink_contract={
            "source_sha256": matrix["input_pins"]["sol_shrink_provenance"][
                "sha256"
            ],
            "native_env_hash": native_env_hash,
            "rows_by_slug": {},
        },
        runnability_contract=run_contract,
        native_env_contract={
            "source_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
            "native_env_hash": native_env_hash,
            "components": copy.deepcopy(fingerprint["components"]),
        },
    )
    live_fingerprint_path = sr.SOL_NATIVE_ENV_FINGERPRINT
    live_fingerprint_payload = live_fingerprint_path.read_bytes()
    live_fingerprint = json.loads(live_fingerprint_payload)
    live_native_env_hash = live_fingerprint["native_env_hash"]
    invocation["adapter_contract"]["environment"] = copy.deepcopy(
        sol_chunk.workload_matrix.EXACT_SOL_ENVIRONMENT
    )
    invocation["adapter_contract"]["native_env_hash"] = live_native_env_hash
    invocation["adapter_contract"]["flashinfer_corpus"] = copy.deepcopy(
        run_contract["flashinfer_corpus_by_category"][category]
    )
    invocation["logical_workload"]["evaluation_evidence"][
        "native_env_hash"
    ] = live_native_env_hash
    invocation["logical_workload"]["evaluation_evidence"].update({
        "source_path": evidence_path.relative_to(cc.REPO_ROOT).as_posix(),
        "source_sha256": bcm.sha256_file(evidence_path),
    })
    sealed = next(
        row for row in matrix["cells"] if row["identity"] == identity
    )
    cell = {
        "identity": identity,
        "authority_cell_sha256": bcm.canonical_sha256(authority_cell),
        "disposition": "runnable",
        "capture_readiness": copy.deepcopy(sealed["capture_readiness"]),
        "trace_budget": copy.deepcopy(sealed["trace_budget"]),
        "invocation": invocation,
        "control_bits": control,
    }
    matrix["input_pins"]["sol_native_env_fingerprint"] = {
        "path": sr.SOL_NATIVE_ENV_FINGERPRINT.relative_to(
            cc.REPO_ROOT
        ).as_posix(),
        "sha256": hashlib.sha256(live_fingerprint_payload).hexdigest(),
        "native_env_hash": live_native_env_hash,
    }
    cell["trace_budget"] = {
        "source_path": "workloads/simbench_bw/trace_budgets.toml",
        "source_sha256": "0" * 64,
        "status": "declared",
        "max_trace_bytes": 100,
        "max_wall_time_s": 10,
        "sampling_policy": "off",
    }
    return matrix, cell


def _fixture_only_launch_set_plan(tmp_path, *, cardinality=3):
    matrix, cell = _generated_sol_cell()
    pins = json.loads(cc.DEFAULT_PINS.read_text())
    pins["toolchain"]["nvcc_release"] = "13.2"
    invocation = cell["invocation"]
    binary = cc.REPO_ROOT / invocation["executable_path"]
    cell_dir = tmp_path / "must-not-exist"
    plan = cc.build_fixture_only_launch_set_plan(
        cell=cell, binary=binary, invocation=invocation,
        cell_dir=cell_dir, matrix=matrix, pins=pins,
        trace_inventory_cardinality=cardinality,
    )
    return plan, matrix, cell, pins, binary, cell_dir


def test_fixture_only_launch_set_plan_has_exact_command_and_artifacts(tmp_path):
    plan, matrix, cell, pins, binary, cell_dir = (
        _fixture_only_launch_set_plan(tmp_path)
    )
    invocation = cell["invocation"]
    identity = cell["identity"]
    shape = invocation["logical_workload"]["value"]
    metrics = matrix["metric_registry"]["collection"][
        "selected_unit_metrics"
    ]
    expected = [
        str(cc.LABEL_DRIVER), "--device", matrix["arch"],
        "--slug", identity["slug"],
        "--shape-point", shape,
        "--shape-binding", "logical-workload",
        "--repeat", "0", "--launch", "0",
        "--binary", str(binary),
        "--warmup", "1",
        "--target-processes", "all",
        "--filter-mode", "global",
        "--replay-mode", "kernel",
        "--cache-control", "all",
        "--clock-control", "none",
        "--profile-from-start", "off",
        "--launch-skip", "0",
        "--launch-count", "4",
        "--out-dir", str(cell_dir / "labels"),
        "--expected-cuda-release", "13.2",
        "--expected-ncu-version-prefix",
        pins["toolchain"]["ncu_version_prefix"],
        "--publication-metric-scope", "cycles_only_binding",
    ]
    for metric in metrics:
        expected += ["--unit-metric", metric]
    expected += ["--", *invocation["label_app_args"]]
    assert plan["command"] == expected
    assert plan["trace_inventory_cardinality"] == 3
    assert plan["driver_selector_count"] == 4
    assert plan["ncu_launch_protocol"] == {
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_replay_mode": "kernel",
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": 0,
        "ncu_launch_count": 4,
    }
    for _field, option in cc._EXPLICIT_NCU_DRIVER_FIELDS:
        assert plan["command"].count(option) == 1
    separator = plan["command"].index("--")
    assert plan["command"][separator + 1:] == invocation["label_app_args"]
    observed_metrics = [
        plan["command"][index + 1]
        for index, value in enumerate(plan["command"])
        if value == "--unit-metric"
    ]
    assert observed_metrics == metrics

    assert [path.name for path in plan["fixed_artifacts"]] == [
        "run_0_launch_set.ncu-rep",
        "run_0_launch_set.details.csv",
        "run_0_launch_set.details.stderr",
        "run_0_launch_set.launch_metrics.json",
        "run_0_launch_set.details_export_receipt.json",
        "run_0_launch_set.launch_inventory.json",
        "run_0_launch_set.execution_identity.json",
        "label_run_manifest_0_launch_set.json",
        "label_launch_set_capture_manifest.json",
    ]
    assert [path.name for path in plan["label_sidecars"]] == [
        "label_0_0.json", "label_0_1.json", "label_0_2.json",
    ]
    assert all(path.parent == plan["cell_root"] for path in (
        *plan["fixed_artifacts"], *plan["label_sidecars"],
    ))
    assert "run_0_0.ncu-rep" not in {
        path.name for path in plan["fixed_artifacts"]
    }
    assert not cell_dir.exists()


@pytest.mark.parametrize(
    "cardinality",
    [None, True, False, "3", 0, -1, cc.ncu_capture_contract.MAX_NCU_SELECTOR],
)
def test_fixture_only_launch_set_plan_rejects_invalid_cardinality(
    tmp_path, cardinality,
):
    matrix, cell = _generated_sol_cell()
    pins = json.loads(cc.DEFAULT_PINS.read_text())
    pins["toolchain"]["nvcc_release"] = "13.2"
    invocation = cell["invocation"]
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.build_fixture_only_launch_set_plan(
            cell=cell,
            binary=cc.REPO_ROOT / invocation["executable_path"],
            invocation=invocation, cell_dir=tmp_path / "absent",
            matrix=matrix, pins=pins,
            trace_inventory_cardinality=cardinality,
        )
    assert caught.value.code == (
        "label_launch_set_inventory_cardinality_invalid"
    )
    assert not (tmp_path / "absent").exists()


def test_fixture_only_launch_set_plan_accepts_upper_selector_boundary(tmp_path):
    cardinality = cc.ncu_capture_contract.MAX_NCU_SELECTOR - 1
    plan, *_rest = _fixture_only_launch_set_plan(
        tmp_path, cardinality=cardinality,
    )
    assert plan["driver_selector_count"] == (
        cc.ncu_capture_contract.MAX_NCU_SELECTOR
    )
    sidecars = plan["label_sidecars"]
    assert len(sidecars) == cardinality
    assert sidecars[0].name == "label_0_0.json"
    assert sidecars[-1].name == f"label_0_{cardinality - 1}.json"
    full_slice = sidecars[:]
    assert isinstance(full_slice, cc._PlannedLabelSidecars)
    assert len(full_slice) == cardinality
    assert full_slice[0] == sidecars[0]
    assert full_slice[-1] == sidecars[-1]
    with pytest.raises(TypeError):
        sidecars[0.0]


def test_fixture_only_launch_set_plan_accepts_one_inventory_row(tmp_path):
    plan, *_rest = _fixture_only_launch_set_plan(tmp_path, cardinality=1)
    assert plan["driver_selector_count"] == 2
    assert [path.name for path in plan["label_sidecars"]] == [
        "label_0_0.json",
    ]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_fixture_only_launch_set_plan_reuses_metric_contract(
    tmp_path, mutation,
):
    matrix, cell = _generated_sol_cell()
    pins = json.loads(cc.DEFAULT_PINS.read_text())
    pins["toolchain"]["nvcc_release"] = "13.2"
    metrics = matrix["metric_registry"]["collection"][
        "selected_unit_metrics"
    ]
    if mutation == "missing":
        metrics.pop()
    elif mutation == "duplicate":
        metrics[-1] = metrics[0]
    else:
        metrics[-1] = "synthetic__unknown.sum"
    invocation = cell["invocation"]
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.build_fixture_only_launch_set_plan(
            cell=cell,
            binary=cc.REPO_ROOT / invocation["executable_path"],
            invocation=invocation, cell_dir=tmp_path / "absent",
            matrix=matrix, pins=pins, trace_inventory_cardinality=1,
        )
    assert caught.value.code == "matrix_metric_selection_drift"


def test_fixture_only_launch_set_plan_reuses_toolchain_and_shape_refusals(
    tmp_path,
):
    matrix, cell = _generated_sol_cell()
    pins = json.loads(cc.DEFAULT_PINS.read_text())
    pins["toolchain"]["nvcc_release"] = "13.3"
    invocation = cell["invocation"]
    kwargs = {
        "cell": cell,
        "binary": cc.REPO_ROOT / invocation["executable_path"],
        "invocation": invocation,
        "cell_dir": tmp_path / "absent",
        "matrix": matrix,
        "pins": pins,
        "trace_inventory_cardinality": 1,
    }
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.build_fixture_only_launch_set_plan(**kwargs)
    assert caught.value.code == "label_toolchain_contract_invalid"

    pins["toolchain"]["nvcc_release"] = "13.2"
    invocation["label_app_args"] += ["--shape-point", "forged"]
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.build_fixture_only_launch_set_plan(**kwargs)
    assert caught.value.code == "sol_shape_token_forwarded"


def test_fixture_only_launch_set_plan_is_pure(tmp_path, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("fixture-only launch-set planning executed a process")

    monkeypatch.setattr(cc, "_run_logged", unexpected)
    monkeypatch.setattr(cc.subprocess, "run", unexpected)
    plan, *_rest, cell_dir = _fixture_only_launch_set_plan(tmp_path)
    assert plan["command"]
    assert not cell_dir.exists()


def test_fixture_only_launch_set_plan_types_control_contract_failure(
    tmp_path, monkeypatch,
):
    def refuse(_protocol):
        raise ValueError("synthetic protocol refusal")

    monkeypatch.setattr(
        cc.ncu_capture_contract, "validate_measurement_protocol", refuse,
    )
    with pytest.raises(cc.ChunkRefusal) as caught:
        _fixture_only_launch_set_plan(tmp_path)
    assert caught.value.code == "label_launch_set_control_contract_invalid"


def _certified_sol_env(cell: dict) -> dict[str, str]:
    """Construct the exact runner-facing environment without probing a GPU."""
    environment = cell["invocation"]["adapter_contract"]["environment"]
    result = {
        key: value for key, value in cc.os.environ.items()
        if key not in {
            *CERTIFYING_SOL_ENVIRONMENT,
            "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
            *cc.sol_env.CAPTURE_CONTROL_VARIABLES,
        }
    }
    cutlass = (cc.REPO_ROOT / environment["CUTLASS_DIR"]).resolve()
    path_prepend = (cc.REPO_ROOT / environment["PATH_prepend"]).resolve()
    cuda_bin = str((Path(environment["CUDA_HOME"]) / "bin").resolve())
    prior_path = [
        value for value in result.get("PATH", "").split(cc.os.pathsep)
        if value and Path(value).resolve() not in {
            Path(cuda_bin), path_prepend,
        }
    ]
    result.update({
        **CERTIFYING_SOL_ENVIRONMENT,
        "CUTLASS_DIR": str(cutlass),
        "PATH": cc.os.pathsep.join([cuda_bin, str(path_prepend), *prior_path]),
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
    return result


def _legacy_certified_sol_env(cell: dict) -> dict[str, str]:
    environment = cell["invocation"]["adapter_contract"]["environment"]
    path_prepend = (cc.REPO_ROOT / environment["PATH_prepend"]).resolve()
    prior_path = [
        value for value in cc.os.environ.get("PATH", "").split(cc.os.pathsep)
        if value and Path(value).resolve() != path_prepend
    ]
    return {
        "NVCC_CCBIN": "g++-14",
        "CUTLASS_DIR": str(
            (cc.REPO_ROOT / environment["CUTLASS_DIR"]).resolve()
        ),
        "PATH": cc.os.pathsep.join([str(path_prepend), *prior_path]),
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }


def _write_valid_sol_stage(stage: Path, cell: dict, invocation: dict) -> dict:
    """Write runner-facing SOL preparation evidence without compiling or a GPU."""
    stage.mkdir(parents=True)
    logical = invocation["logical_workload"]
    source_paths = {
        "definition": (cc.REPO_ROOT / logical["definition_file"]).resolve(),
        "workload": (cc.REPO_ROOT / logical["workload_file"]).resolve(),
        "solution": (
            cc.REPO_ROOT / invocation["authority_reference_path"]
        ).resolve(),
    }
    source_rows = [
        row for row in source_paths["workload"].read_bytes().splitlines()
        if row.strip()
    ]
    source_index = invocation["capture_target"]["source_workload_index"]
    selected_row = source_rows[source_index]
    selected_workload = json.loads(selected_row)
    definition_payload = json.loads(source_paths["definition"].read_text())
    solution_payload = json.loads(source_paths["solution"].read_text())
    artifact_payloads = {
        "definition": source_paths["definition"].read_bytes(),
        "workload": selected_row + b"\n",
        "solution": source_paths["solution"].read_bytes(),
        "eval_driver": b"# synthetic no-GPU test driver\n",
        "capture_runtime": b"# synthetic no-GPU runtime\n",
        "benchmark_kernel_so": b"synthetic benchmark module\n",
    }
    artifact_relpaths = {
        "definition": "definition.json",
        "workload": "workload.jsonl",
        "solution": "solution.json",
        "eval_driver": "eval_driver.py",
        "capture_runtime": "_capture_runtime.py",
        "benchmark_kernel_so": "benchmark_kernel.so",
    }
    artifacts = {}
    for name, payload in artifact_payloads.items():
        path = stage / artifact_relpaths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts[name] = {
            "path": artifact_relpaths[name],
            "sha256": cc.sha256_file(path),
        }
    definition_source_copy = stage / "definition.source.json"
    definition_source_copy.write_bytes(source_paths["definition"].read_bytes())
    artifacts["definition_source"] = {
        "path": "definition.source.json",
        "sha256": cc.sha256_file(definition_source_copy),
    }
    definition_comparison = cc.sol_def.require_definition_equivalent(
        definition_source_copy.read_bytes(),
        (stage / "definition.json").read_bytes(),
    )
    definition_attestation = definition_comparison.attestation
    definition_attestation_path = (
        stage / "definition_equivalence_attestation.json"
    )
    definition_attestation_path.write_bytes(
        cc.sol_def.serialize_attestation(definition_attestation)
    )
    artifacts["definition_equivalence_attestation"] = {
        "path": "definition_equivalence_attestation.json",
        "sha256": cc.sha256_file(definition_attestation_path),
    }
    adapter = (cc.REPO_ROOT / invocation["executable_path"]).resolve()
    artifacts["adapter"] = {
        "path": str(adapter),
        "sha256": invocation["executable_sha256"],
    }
    adapter_source = (
        cc.REPO_ROOT / "tools/sol_execbench/src/sol_execbench/driver"
        / "capture_adapter.py"
    ).resolve()
    certified_env_source = adapter_source.with_name("certified_env.py")
    artifacts["adapter_source"] = {
        "path": str(adapter_source),
        "sha256": cc.sha256_file(adapter_source),
    }
    artifacts["certified_env_source"] = {
        "path": str(certified_env_source),
        "sha256": cc.sha256_file(certified_env_source),
    }
    solution_sources = []
    for source in solution_payload["sources"]:
        relative = source["path"]
        content = source.get("content")
        if not content:
            content = (source_paths["solution"].parent / relative).read_text()
        source_path = stage / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(content)
        solution_sources.append({
            "relative_path": relative,
            "path": relative,
            "sha256": cc.sha256_file(source_path),
        })
    artifacts["solution_sources"] = solution_sources
    environment_contract, environment_contract_sha256 = (
        cc._sol_certifying_environment(invocation)
    )
    flashinfer_contract, flashinfer_contract_sha256, _runtime_root = (
        cc._sol_flashinfer_corpus_contract(invocation)
    )
    manifest = {
        "schema_version": 1,
        "manifest_kind": "sol_execbench_capture_preparation",
        "capture_contract_version": 2,
        "path_mode": "stage_relative_v1",
        "stage_root": ".",
        "source_inputs": {
            "definition": {
                "path": str(source_paths["definition"]),
                "sha256": logical["definition_file_sha256"],
            },
            "workload": {
                "path": str(source_paths["workload"]),
                "sha256": logical["workload_file_sha256"],
            },
            "solution": {
                "path": str(source_paths["solution"]),
                "sha256": invocation["authority_reference_sha256"],
            },
        },
        "selection": {
            "source_workload_index": source_index,
            "source_workload_count": len(source_rows),
            "staged_workload_index": 0,
            "workload_uuid": selected_workload["uuid"],
            "workload_axes": selected_workload["axes"],
            "source_row_sha256": hashlib.sha256(selected_row).hexdigest(),
        },
        "solution_contract": {
            "definition": definition_payload["name"],
            "solution": solution_payload["name"],
            "languages": solution_payload["spec"]["languages"],
            "entry_point": solution_payload["spec"]["entry_point"],
            "destination_passing_style": solution_payload["spec"][
                "destination_passing_style"
            ],
        },
        "benchmark_config": {
            "warmup_runs": 10,
            "iterations": 50,
            "lock_clocks": False,
            "benchmark_reference": True,
            "seed": 200,
        },
        "environment_contract": environment_contract,
        "environment_contract_sha256": environment_contract_sha256,
        "flashinfer_corpus_contract": flashinfer_contract,
        "flashinfer_corpus_contract_sha256": flashinfer_contract_sha256,
        "source_definition_sha256": definition_comparison.source_sha256,
        "staged_definition_sha256": definition_comparison.staged_sha256,
        "definition_equivalence": definition_attestation,
        "artifacts": artifacts,
        "execution": {
            "command": [str(Path(cc.sys.executable).resolve()), "eval_driver.py"],
            "working_directory": ".",
            "timing_methodology": "cuda_events",
            "correctness_rounds": 1,
            "warmup_calls": 1,
            "profiled_candidate_calls": 1,
        },
    }
    manifest["stage_definition_sha256"] = cc._sol_manifest_sha256(manifest)
    (stage / "capture_prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _reseal_sol_stage_manifest(manifest: dict) -> None:
    unsigned = {
        key: value for key, value in manifest.items()
        if key != "stage_definition_sha256"
    }
    manifest["stage_definition_sha256"] = cc._sol_manifest_sha256(unsigned)


def _prepare_sol_context(tmp_path: Path, monkeypatch):
    matrix, cell = _generated_sol_cell()
    invocation = cell["invocation"]
    binary = (cc.REPO_ROOT / invocation["executable_path"]).resolve()
    cell_dir = tmp_path / "attempt" / "cells" / "sol_execbench__problem_001"
    cell_dir.mkdir(parents=True)
    observed = {}

    def fake_run(
        command, *, cwd, env, timeout_s, log_dir, stem, failure_code,
        trusted_child_storage_marker=False,
    ):
        observed.update({
            "command": command, "cwd": cwd, "env": env,
            "timeout_s": timeout_s, "failure_code": failure_code,
        })
        request_path = Path(env["SOL_EXECBENCH_CAPTURE_REQUEST"])
        request = json.loads(request_path.read_text())
        _write_valid_sol_stage(Path(request["stage_dir"]), cell, invocation)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = log_dir / f"{stem}.stdout.log"
        stderr = log_dir / f"{stem}.stderr.log"
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        return 0.1, stdout, stderr

    monkeypatch.setattr(cc, "_run_logged", fake_run)
    context = cc.prepare_sol_capture(
        cell=cell, binary=binary, invocation=invocation, cell_dir=cell_dir,
        runtime_cwd=cc.REPO_ROOT,
        runtime_env={**_certified_sol_env(cell), "BASE_MARKER": "preserved"},
        timeout_s=9,
    )
    return matrix, cell, invocation, binary, cell_dir, context, observed


def _write_valid_sol_phase_evidence(
        context: dict, env: dict[str, str]) -> tuple[Path, Path, dict, dict]:
    request = json.loads(
        Path(env["SOL_EXECBENCH_CAPTURE_REQUEST"]).read_text()
    )
    selection_path = context["stage"] / request["selection_manifest"]
    result_path = context["stage"] / request["result_manifest"]
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": 1,
        "manifest_kind": "sol_execbench_capture_selection",
        "capture_contract_version": 2,
        "capture_kind": request["capture_kind"],
        "path_mode": "stage_relative_v1",
        "stage_root": ".",
        "prepare_manifest": {
            "path": "capture_prepare_manifest.json",
            "sha256": context["manifest_sha256"],
        },
        "source_workload_index": request["source_workload_index"],
        "staged_workload_index": 0,
        "workload": context["manifest"]["selection"],
        "definition": context["manifest"]["solution_contract"]["definition"],
        "solution": context["manifest"]["solution_contract"]["solution"],
        "environment_contract": copy.deepcopy(context["environment_contract"]),
        "environment_contract_sha256": context[
            "environment_contract_sha256"
        ],
        "flashinfer_corpus_contract": copy.deepcopy(
            context["flashinfer_corpus_contract"]
        ),
        "flashinfer_corpus_contract_sha256": context[
            "flashinfer_corpus_contract_sha256"
        ],
        "stage_definition_sha256": context["stage_definition_sha256"],
        "benchmark_kernel_so": context["manifest"]["artifacts"][
            "benchmark_kernel_so"
        ],
        "eval_driver": context["manifest"]["artifacts"]["eval_driver"],
        "capture_runtime": context["manifest"]["artifacts"][
            "capture_runtime"
        ],
        "result_manifest": request["result_manifest"],
    }
    selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "manifest_kind": "sol_execbench_capture_result",
        "capture_contract_version": 2,
        "capture_kind": request["capture_kind"],
        "status": "captured",
        "path_mode": "stage_relative_v1",
        "stage_root": ".",
        "selection_manifest": {
            "path": request["selection_manifest"],
            "sha256": cc.sha256_file(selection_path),
        },
        "prepare_manifest": {
            "path": "capture_prepare_manifest.json",
            "sha256": context["manifest_sha256"],
        },
        "definition": context["manifest"]["solution_contract"]["definition"],
        "solution": context["manifest"]["solution_contract"]["solution"],
        "source_workload_index": request["source_workload_index"],
        "staged_workload_index": 0,
        "workload_uuid": context["manifest"]["selection"]["workload_uuid"],
        "workload_axes": context["manifest"]["selection"]["workload_axes"],
        "environment_contract": copy.deepcopy(context["environment_contract"]),
        "environment_contract_sha256": context[
            "environment_contract_sha256"
        ],
        "flashinfer_corpus_contract": copy.deepcopy(
            context["flashinfer_corpus_contract"]
        ),
        "flashinfer_corpus_contract_sha256": context[
            "flashinfer_corpus_contract_sha256"
        ],
        "stage_definition_sha256": context["stage_definition_sha256"],
        "benchmark_kernel_so": dict(
            context["manifest"]["artifacts"]["benchmark_kernel_so"]
        ),
        "execution_order": [
            "correctness_outside_roi",
            "warmup_outside_roi",
            "cuda_synchronize",
            "cudaProfilerStart",
            "candidate_user_fn_once",
            "cuda_synchronize",
            "cudaProfilerStop_finally",
        ],
        "roi": {
            "profiler_start_calls": 1,
            "candidate_user_fn_calls": 1,
            "profiler_stop_calls": 1,
            "timing_methodology": "cuda_events",
        },
        "outside_roi": {
            "correctness_rounds": 1,
            "correctness_user_fn_calls": 1,
            "warmup_user_fn_calls": 1,
            "all_completed_before_profiler_start": True,
        },
        "correctness": {"status": "passed"},
    }
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    return selection_path, result_path, selection, result


def test_prepare_sol_capture_binds_exact_request_and_survives_parent_rename(
        tmp_path, monkeypatch):
    _, cell, invocation, binary, cell_dir, context, observed = (
        _prepare_sol_context(tmp_path, monkeypatch)
    )
    request_path = Path(
        observed["env"]["SOL_EXECBENCH_CAPTURE_REQUEST"]
    )
    request = json.loads(request_path.read_text())
    assert request_path.is_absolute()
    assert request == {
        "schema_version": 1,
        "action": "prepare",
        "stage_dir": str((cell_dir / "sol_stage").resolve()),
        "source_workload_index": invocation["capture_target"][
            "source_workload_index"
        ],
        "compile_timeout_s": 9,
        "environment_contract": context["environment_contract"],
        "environment_contract_sha256": context[
            "environment_contract_sha256"
        ],
        "flashinfer_corpus_contract": context[
            "flashinfer_corpus_contract"
        ],
        "flashinfer_corpus_contract_sha256": context[
            "flashinfer_corpus_contract_sha256"
        ],
    }
    assert observed["command"] == [str(binary), *invocation["trace_app_args"]]
    assert observed["cwd"] == cc.REPO_ROOT
    assert observed["env"]["BASE_MARKER"] == "preserved"
    assert observed["timeout_s"] == 9
    assert observed["failure_code"] == "sol_capture_prepare_failed"
    assert context["stage"] == cell_dir / "sol_stage"
    assert context["benchmark_kernel_so"].is_file()
    assert context["manifest_sha256"] == cc.sha256_file(
        context["manifest_path"]
    )

    old_parent = tmp_path / "attempt"
    sealed_parent = tmp_path / "sealed" / "chunk-000"
    sealed_parent.parent.mkdir()
    old_parent.rename(sealed_parent)
    moved_stage = sealed_parent / "cells" / cell_dir.name / "sol_stage"
    revalidated = cc._validate_sol_prepare_manifest(
        cell=cell, invocation=invocation, stage=moved_stage,
    )
    assert revalidated["manifest_sha256"] == context["manifest_sha256"]
    serialized = json.dumps(revalidated["manifest"], sort_keys=True)
    assert str(old_parent) not in serialized
    assert str(sealed_parent) not in serialized


def test_problem070_real_packager_stage_passes_runner_v2_validation(tmp_path):
    """Binding P3/P12 integration: real serialization, no compiler/GPU."""
    _matrix_document, cell = _generated_sol_cell("problem_070_")
    invocation = cell["invocation"]
    logical = invocation["logical_workload"]
    definition = cc._resolve_repo_path(
        logical["definition_file"], "SOL definition"
    )
    workload = cc._resolve_repo_path(
        logical["workload_file"], "SOL workload"
    )
    solution = cc._resolve_repo_path(
        invocation["authority_reference_path"], "SOL solution"
    )
    benchmark_dir = cc._resolve_repo_path(
        logical["benchmark_dir"], "SOL benchmark directory"
    )
    config = benchmark_dir / "config.json"
    executable = cc._resolve_repo_path(
        invocation["executable_path"], "SOL adapter"
    )
    assert cc.sha256_file(definition) == (
        "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa"
    )
    assert "op_type" not in json.loads(definition.read_text())

    stage = tmp_path / "problem-070-stage"
    script = r"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from sol_execbench.driver import capture_adapter as adapter
from sol_execbench.driver import problem_packager

definition, workload, solution, stage, index, config, executable, corpus_json = sys.argv[1:]
problem_packager._get_local_sm = lambda: "sm_86"

class NoGpuCompiler:
    calls = 0
    def __call__(self, _command, *, cwd, **_kwargs):
        self.calls += 1
        (Path(cwd) / "benchmark_kernel.so").write_bytes(
            b"synthetic compiled module; never loaded\n"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

compiler = NoGpuCompiler()
outcome = adapter.prepare_capture_stage(
    definition_file=Path(definition),
    workload_file=Path(workload),
    solution_file=Path(solution),
    staging_dir=Path(stage),
    source_workload_index=int(index),
    config_file=Path(config) if config else None,
    adapter_executable=Path(executable),
    environment_contract={
        "CUDA_HOME": "/usr/local/cuda-13.2",
        "NVCC_CCBIN": "g++-14",
        "CC": "gcc-14",
        "CXX": "g++-14",
        "CUDAHOSTCXX": "g++-14",
    },
    flashinfer_corpus_contract=json.loads(corpus_json),
    environ={"PATH": os.environ["PATH"]},
    run=compiler,
)
assert compiler.calls == 1
print(json.dumps({
    "source": outcome["manifest"]["source_definition_sha256"],
    "staged": outcome["manifest"]["staged_definition_sha256"],
}))
"""
    outcome = subprocess.run(
        [
            str(
                cc.REPO_ROOT
                / "tools" / "sol_execbench" / ".venv" / "bin" / "python"
            ),
            "-c", script, str(definition), str(workload), str(solution),
            str(stage),
            str(invocation["capture_target"]["source_workload_index"]),
            str(config) if config.is_file() else "", str(executable),
            json.dumps(invocation["adapter_contract"]["flashinfer_corpus"]),
        ],
        cwd=cc.REPO_ROOT,
        env={
            **cc.os.environ,
            "PYTHONPATH": cc.os.pathsep.join([
                str(cc.REPO_ROOT),
                str(cc.REPO_ROOT / "tools" / "sol_execbench" / "src"),
            ]),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert outcome.returncode == 0, outcome.stderr
    staged_pins = json.loads(outcome.stdout)
    assert staged_pins["source"] == (
        "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa"
    )
    assert staged_pins["staged"] == (
        "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387"
    )
    evidence = cc._validate_sol_prepare_manifest(
        cell=cell, invocation=invocation, stage=stage,
    )
    assert evidence["definition_equivalence"]["rule_id"] == (
        cc.sol_def.CERTIFYING_RULE_ID
    )
    assert evidence["definition_equivalence"][
        "top_level_field_states"
    ]["op_type"] == {
        "source": "missing",
        "staged": "null",
        "source_normalized": True,
        "staged_normalized": False,
    }
    assert not any(
        value["source_normalized"] or value["staged_normalized"]
        for value in evidence["definition_equivalence"][
            "axis_description_states"
        ].values()
    )


@pytest.mark.parametrize(
    "tamper,reason",
    [
        ("artifact_hash", "input_hash_mismatch"),
        ("artifact_escape", "sol_stage_manifest_invalid"),
        ("staged_source_substitution", "sol_stage_source_mismatch"),
        ("wrong_selected_row", "sol_stage_workload_row_mismatch"),
        ("source_path", "sol_stage_source_mismatch"),
        ("adapter_path", "sol_stage_adapter_mismatch"),
        ("environment_contract", "sol_stage_environment_contract_mismatch"),
        ("environment_digest", "sol_stage_environment_contract_mismatch"),
        ("certified_env_hash", "input_hash_mismatch"),
        ("solution_source", "sol_stage_solution_sources_invalid"),
        ("benchmark_config", "sol_stage_config_invalid"),
        ("embedded_stage", "sol_stage_manifest_invalid"),
    ],
)
def test_sol_prepare_manifest_refuses_path_hash_and_relocation_tamper(
        tmp_path, tamper, reason):
    _, cell = _generated_sol_cell()
    invocation = cell["invocation"]
    stage = tmp_path / "stage"
    manifest = _write_valid_sol_stage(stage, cell, invocation)
    if tamper == "artifact_hash":
        manifest["artifacts"]["benchmark_kernel_so"]["sha256"] = "0" * 64
    elif tamper == "artifact_escape":
        manifest["artifacts"]["benchmark_kernel_so"]["path"] = "../escape.so"
    elif tamper == "staged_source_substitution":
        staged_definition = stage / manifest["artifacts"]["definition"]["path"]
        staged_definition.write_bytes(b'{"forged":true}\n')
        manifest["artifacts"]["definition"]["sha256"] = cc.sha256_file(
            staged_definition
        )
    elif tamper == "wrong_selected_row":
        source_workload = Path(manifest["source_inputs"]["workload"]["path"])
        rows = [
            row for row in source_workload.read_bytes().splitlines()
            if row.strip()
        ]
        wrong_row = rows[1]
        staged_workload = stage / manifest["artifacts"]["workload"]["path"]
        staged_workload.write_bytes(wrong_row + b"\n")
        manifest["artifacts"]["workload"]["sha256"] = cc.sha256_file(
            staged_workload
        )
        manifest["selection"]["source_row_sha256"] = hashlib.sha256(
            wrong_row
        ).hexdigest()
    elif tamper == "source_path":
        manifest["source_inputs"]["definition"]["path"] = (
            manifest["source_inputs"]["solution"]["path"]
        )
    elif tamper == "adapter_path":
        manifest["artifacts"]["adapter"]["path"] = "/bin/true"
    elif tamper == "environment_contract":
        manifest["environment_contract"]["NVCC_CCBIN"] = "g++-15"
    elif tamper == "environment_digest":
        manifest["environment_contract_sha256"] = "0" * 64
    elif tamper == "certified_env_hash":
        manifest["artifacts"]["certified_env_source"]["sha256"] = "0" * 64
    elif tamper == "solution_source":
        entry = manifest["artifacts"]["solution_sources"][0]
        source_path = stage / entry["path"]
        source_path.write_text(source_path.read_text() + "\n// forged\n")
        entry["sha256"] = cc.sha256_file(source_path)
    elif tamper == "benchmark_config":
        manifest["benchmark_config"]["seed"] += 1
    elif tamper == "embedded_stage":
        manifest["stage_dir"] = str(stage)
    _reseal_sol_stage_manifest(manifest)
    (stage / "capture_prepare_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    with pytest.raises(cc.ChunkRefusal, match=reason):
        cc._validate_sol_prepare_manifest(
            cell=cell, invocation=invocation, stage=stage,
        )


def test_sol_prepare_manifest_refuses_stale_stage_definition_digest(tmp_path):
    _, cell = _generated_sol_cell()
    invocation = cell["invocation"]
    stage = tmp_path / "stage"
    manifest = _write_valid_sol_stage(stage, cell, invocation)
    manifest["stage_definition_sha256"] = "0" * 64
    (stage / "capture_prepare_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    with pytest.raises(cc.ChunkRefusal, match="sol_stage_definition_hash_mismatch"):
        cc._validate_sol_prepare_manifest(
            cell=cell, invocation=invocation, stage=stage,
        )


def _rewrite_sol_attestation(stage: Path, manifest: dict) -> None:
    attestation_path = stage / "definition_equivalence_attestation.json"
    attestation_path.write_bytes(
        cc.sol_def.serialize_attestation(manifest["definition_equivalence"])
    )
    manifest["artifacts"]["definition_equivalence_attestation"][
        "sha256"
    ] = cc.sha256_file(attestation_path)
    _reseal_sol_stage_manifest(manifest)
    (stage / "capture_prepare_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )


def test_sol_prepare_manifest_refuses_v1_attestation_rule(tmp_path):
    _, cell = _generated_sol_cell()
    invocation = cell["invocation"]
    stage = tmp_path / "stage"
    manifest = _write_valid_sol_stage(stage, cell, invocation)
    manifest["definition_equivalence"]["rule_id"] = "SOL_DEF_EQUIV_V1"
    _reseal_sol_stage_manifest(manifest)
    (stage / "capture_prepare_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )

    with pytest.raises(
        cc.ChunkRefusal, match="SOL_ATTESTATION_RULE_MISMATCH"
    ):
        cc._validate_sol_prepare_manifest(
            cell=cell, invocation=invocation, stage=stage,
        )


def test_sol_prepare_manifest_refuses_attestation_state_drift(tmp_path):
    _, cell = _generated_sol_cell()
    invocation = cell["invocation"]
    stage = tmp_path / "stage"
    manifest = _write_valid_sol_stage(stage, cell, invocation)
    state = manifest["definition_equivalence"]["top_level_field_states"][
        "hf_id"
    ]
    state["source"] = "null"
    state["source_normalized"] = False
    _rewrite_sol_attestation(stage, manifest)

    with pytest.raises(
        cc.ChunkRefusal, match="SOL_ATTESTATION_STATE_MISMATCH"
    ):
        cc._validate_sol_prepare_manifest(
            cell=cell, invocation=invocation, stage=stage,
        )


def test_sol_prepare_manifest_refuses_ineligible_axis_state_pointer(tmp_path):
    _, cell = _generated_sol_cell()
    invocation = cell["invocation"]
    stage = tmp_path / "stage"
    manifest = _write_valid_sol_stage(stage, cell, invocation)
    manifest["definition_equivalence"]["axis_description_states"][
        "not_in_either_definition"
    ] = {
        "source": "null",
        "staged": "null",
        "source_normalized": False,
        "staged_normalized": False,
    }
    _rewrite_sol_attestation(stage, manifest)

    with pytest.raises(
        cc.ChunkRefusal, match="SOL_ATTESTATION_POINTER_INELIGIBLE"
    ):
        cc._validate_sol_prepare_manifest(
            cell=cell, invocation=invocation, stage=stage,
        )


def test_sol_prepare_manifest_refuses_ineligible_tensor_state_pointer(tmp_path):
    _, cell = _generated_sol_cell()
    invocation = cell["invocation"]
    stage = tmp_path / "stage"
    manifest = _write_valid_sol_stage(stage, cell, invocation)
    manifest["definition_equivalence"]["tensor_description_states"][
        "/outputs/not_in_either_definition/description"
    ] = {
        "source": "null",
        "staged": "null",
        "source_normalized": False,
        "staged_normalized": False,
    }
    _rewrite_sol_attestation(stage, manifest)

    with pytest.raises(
        cc.ChunkRefusal, match="SOL_ATTESTATION_POINTER_INELIGIBLE"
    ):
        cc._validate_sol_prepare_manifest(
            cell=cell, invocation=invocation, stage=stage,
        )


def test_sol_prepare_manifest_rehashes_adapter_after_execution(tmp_path):
    _, cell = _generated_sol_cell()
    invocation = copy.deepcopy(cell["invocation"])
    adapter = tmp_path / "sol-execbench"
    adapter.write_bytes(b"pinned adapter\n")
    adapter.chmod(0o755)
    invocation["executable_path"] = str(adapter.resolve())
    invocation["executable_sha256"] = cc.sha256_file(adapter)
    stage = tmp_path / "stage"
    _write_valid_sol_stage(stage, cell, invocation)
    adapter.write_bytes(b"adapter changed after launch\n")
    with pytest.raises(cc.ChunkRefusal, match="input_hash_mismatch"):
        cc._validate_sol_prepare_manifest(
            cell=cell, invocation=invocation, stage=stage,
        )


def test_sol_phase_env_and_result_are_exact_hashed_and_stage_relative(
        tmp_path, monkeypatch):
    _, _, invocation, _, _, context, _ = _prepare_sol_context(
        tmp_path, monkeypatch
    )
    env = cc.sol_phase_env(
        context, phase_tag="trace/full#1", capture_kind="trace", timeout_s=8
    )
    request_path = Path(env["SOL_EXECBENCH_CAPTURE_REQUEST"])
    request = json.loads(request_path.read_text())
    assert request_path == context["request_dir"] / "trace_full_1.json"
    assert request == {
        "schema_version": 1,
        "action": "execute",
        "stage_dir": str(context["stage"].resolve()),
        "source_workload_index": invocation["capture_target"][
            "source_workload_index"
        ],
        "capture_kind": "trace",
        "selection_manifest": "evidence/trace_full_1-selection.json",
        "result_manifest": "evidence/trace_full_1-result.json",
        "execution_timeout_s": 8,
        "environment_contract": context["environment_contract"],
        "environment_contract_sha256": context[
            "environment_contract_sha256"
        ],
        "flashinfer_corpus_contract": context[
            "flashinfer_corpus_contract"
        ],
        "flashinfer_corpus_contract_sha256": context[
            "flashinfer_corpus_contract_sha256"
        ],
    }
    assert env["BASE_MARKER"] == "preserved"
    selection_path, result_path, selection, result = (
        _write_valid_sol_phase_evidence(context, env)
    )
    evidence = cc.validate_sol_phase_env(context, env, "trace")
    assert evidence == {
        "capture_kind": "trace",
        "selection_relative_path": request["selection_manifest"],
        "selection_sha256": cc.sha256_file(selection_path),
        "result_relative_path": request["result_manifest"],
        "result_sha256": cc.sha256_file(result_path),
        "prepare_manifest_sha256": context["manifest_sha256"],
        "benchmark_kernel_sha256": context["benchmark_kernel_sha256"],
        "roi": result["roi"],
        "outside_roi": result["outside_roi"],
    }
    serialized = json.dumps({"selection": selection, "result": result})
    assert str(context["stage"]) not in serialized
    with pytest.raises(cc.ChunkRefusal, match="capture_evidence_exists"):
        cc.sol_phase_env(
            context, phase_tag="trace/full#1", capture_kind="trace", timeout_s=8
        )


@pytest.mark.parametrize(
    "tamper,reason",
    [
        ("selection_hash", "sol_phase_result_invalid"),
        ("prepare_hash", "sol_phase_result_invalid"),
        ("module_hash", "sol_phase_module_mismatch"),
        ("module_path", "sol_phase_module_mismatch"),
        ("selection_identity", "sol_phase_selection_mismatch"),
        ("selection_environment", "sol_phase_selection_mismatch"),
        ("result_environment", "sol_phase_result_invalid"),
        ("stage_definition", "sol_phase_result_invalid"),
        ("absolute_stage_leak", "sol_phase_not_relocatable"),
        ("roi_count", "sol_phase_roi_invalid"),
    ],
)
def test_sol_phase_validation_refuses_fake_or_mismatched_evidence(
        tmp_path, monkeypatch, tamper, reason):
    _, _, _, _, _, context, _ = _prepare_sol_context(tmp_path, monkeypatch)
    env = cc.sol_phase_env(
        context, phase_tag="label", capture_kind="label", timeout_s=8
    )
    selection_path, result_path, selection, result = (
        _write_valid_sol_phase_evidence(context, env)
    )
    if tamper == "selection_hash":
        result["selection_manifest"]["sha256"] = "0" * 64
    elif tamper == "prepare_hash":
        result["prepare_manifest"]["sha256"] = "0" * 64
    elif tamper == "module_hash":
        result["benchmark_kernel_so"]["sha256"] = "0" * 64
    elif tamper == "module_path":
        result["benchmark_kernel_so"]["path"] = "build/forged-kernel.so"
    elif tamper == "selection_identity":
        selection["source_workload_index"] += 1
    elif tamper == "selection_environment":
        selection["environment_contract"]["NVCC_CCBIN"] = "g++-15"
    elif tamper == "result_environment":
        result["environment_contract_sha256"] = "0" * 64
    elif tamper == "stage_definition":
        result["stage_definition_sha256"] = "0" * 64
    elif tamper == "absolute_stage_leak":
        selection["stage_dir"] = str(context["stage"])
    elif tamper == "roi_count":
        result["roi"]["candidate_user_fn_calls"] = 2
    selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n")
    if tamper not in {"selection_hash", "selection_identity",
                      "selection_environment",
                      "absolute_stage_leak"}:
        result["selection_manifest"]["sha256"] = cc.sha256_file(selection_path)
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    with pytest.raises(cc.ChunkRefusal, match=reason):
        cc.validate_sol_phase_env(context, env, "label")


@pytest.mark.parametrize(
    "tamper",
    [
        "schema", "action", "stage_dir", "source_index",
        "environment_contract", "environment_digest",
        "absolute_evidence_path", "request_path",
    ],
)
def test_sol_phase_validation_refuses_forged_execute_request(
        tmp_path, monkeypatch, tamper):
    _, _, _, _, _, context, _ = _prepare_sol_context(tmp_path, monkeypatch)
    env = cc.sol_phase_env(
        context, phase_tag="control", capture_kind="control", timeout_s=8
    )
    request_path = Path(env["SOL_EXECBENCH_CAPTURE_REQUEST"])
    request = json.loads(request_path.read_text())
    selection_path, result_path, _, result = _write_valid_sol_phase_evidence(
        context, env
    )
    if tamper == "schema":
        request["schema_version"] = 2
        request_path.write_text(json.dumps(request) + "\n")
    elif tamper == "action":
        request["action"] = "prepare"
        request_path.write_text(json.dumps(request) + "\n")
    elif tamper == "stage_dir":
        request["stage_dir"] = str((tmp_path / "forged-stage").resolve())
        request_path.write_text(json.dumps(request) + "\n")
    elif tamper == "source_index":
        request["source_workload_index"] += 1
        request_path.write_text(json.dumps(request) + "\n")
    elif tamper == "environment_contract":
        request["environment_contract"]["NVCC_CCBIN"] = "g++-15"
        request_path.write_text(json.dumps(request) + "\n")
    elif tamper == "environment_digest":
        request["environment_contract_sha256"] = "0" * 64
        request_path.write_text(json.dumps(request) + "\n")
    elif tamper == "absolute_evidence_path":
        request["selection_manifest"] = str(selection_path.resolve())
        request_path.write_text(json.dumps(request) + "\n")
        result["selection_manifest"]["path"] = str(selection_path.resolve())
        result_path.write_text(json.dumps(result) + "\n")
    else:
        forged = tmp_path / "forged-execute-request.json"
        forged.write_text(json.dumps(request) + "\n")
        env = dict(env)
        env["SOL_EXECBENCH_CAPTURE_REQUEST"] = str(forged.resolve())
    with pytest.raises(cc.ChunkRefusal, match="sol_phase_request_invalid"):
        cc.validate_sol_phase_env(context, env, "control")


def test_validate_sol_cell_rederives_workload_solution_and_adapter(monkeypatch):
    _, cell = _generated_sol_cell()
    budget = SimpleNamespace(
        max_trace_bytes=100, max_wall_time_s=10, sampling_policy="off"
    )
    monkeypatch.setattr(cc, "require_cell_budget", lambda *_: budget)
    adapter, invocation, got_budget, cwd, env = cc.validate_sol_cell(
        cell, cell_budgets=object(), certified_sol_env=_certified_sol_env(cell),
    )
    assert adapter == cc.REPO_ROOT / invocation["executable_path"]
    assert got_budget is budget
    assert cwd == cc.REPO_ROOT
    assert Path(env["CUTLASS_DIR"]) == cc.REPO_ROOT / "third_party/cutlass"
    assert env["PATH"].split(":")[:2] == [
        "/usr/local/cuda-13.2/bin",
        str(cc.REPO_ROOT / "tools/sol_execbench/.venv/bin"),
    ]
    assert not cc._shape_tokens(invocation["label_app_args"])


def test_validate_sol_cell_accepts_only_exact_legacy_canonical_environment(
    monkeypatch,
):
    _, cell = _generated_sol_cell()
    environment = cell["invocation"]["adapter_contract"]["environment"]
    environment.clear()
    environment.update(cc.sol_env.LEGACY_MATRIX_ENVIRONMENT)
    budget = SimpleNamespace(
        max_trace_bytes=100, max_wall_time_s=10, sampling_policy="off"
    )
    monkeypatch.setattr(cc, "require_cell_budget", lambda *_: budget)
    _adapter, _invocation, _budget, _cwd, runtime = cc.validate_sol_cell(
        cell,
        cell_budgets=object(),
        certified_sol_env=_legacy_certified_sol_env(cell),
    )
    assert runtime["NVCC_CCBIN"] == "g++-14"
    assert runtime["PATH"].split(cc.os.pathsep)[0] == str(
        cc.REPO_ROOT / "tools/sol_execbench/.venv/bin"
    )
    assert not (
        set(cc.sol_env.PINNED_COMPILER_ENVIRONMENT) - {"NVCC_CCBIN"}
    ).intersection(runtime)

    environment["CC"] = "gcc-14"
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.validate_sol_cell(
            cell,
            cell_budgets=object(),
            certified_sol_env=_legacy_certified_sol_env(cell),
        )
    assert caught.value.code == "SOL_ENV_MATRIX_CONTRACT_INVALID"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda cell: cell["invocation"]["logical_workload"].update(value=""),
         "sol_logical_shape_missing"),
        (lambda cell: cell["invocation"]["logical_workload"].update(
            benchmark_definition="wrong_definition"), "sol_definition_mismatch"),
        (lambda cell: cell["invocation"].update(
            authority_reference_sha256="f" * 64), "input_hash_mismatch"),
        (lambda cell: cell["invocation"]["label_app_args"].extend(
            ["--shape-point", "workloads"]), "sol_shape_token_forwarded"),
    ],
)
def test_validate_sol_cell_refuses_stale_or_fake_contract(monkeypatch, mutation, reason):
    _, cell = _generated_sol_cell()
    budget = SimpleNamespace(
        max_trace_bytes=100, max_wall_time_s=10, sampling_policy="off"
    )
    monkeypatch.setattr(cc, "require_cell_budget", lambda *_: budget)
    mutation(cell)
    with pytest.raises(cc.ChunkRefusal, match=reason):
        cc.validate_sol_cell(
            cell, cell_budgets=object(),
            certified_sol_env=_certified_sol_env(cell),
        )


def test_capture_dispatch_refuses_sol_as_direct_binary(monkeypatch):
    _, cell = _generated_sol_cell()
    cell["invocation"]["adapter_kind"] = "direct_binary"
    with pytest.raises(cc.ChunkRefusal, match="adapter_source_mismatch"):
        cc.validate_capture_cell(cell, cell_budgets=object())


def test_capture_dispatch_threads_retained_runnability_path(monkeypatch, tmp_path):
    retained = tmp_path / "owner" / "sol_runnability.external.v2.json"
    evidence = {"source_kind": "current_full_correctness"}
    binding = {"cohort": "native"}
    sentinel = (
        Path("/bin/true"), {"adapter_kind": "sol_execbench"}, object(),
        cc.REPO_ROOT, {},
    )

    def validate(cell, **kwargs):
        assert cell["identity"]["roster_source"] == "sol_execbench"
        assert kwargs["workload_row_binding"] is binding
        assert kwargs["supplemental_runnability_evidence"] is evidence
        assert kwargs["runnability_path"] == retained
        return sentinel

    monkeypatch.setattr(cc, "validate_sol_cell", validate)
    cell = {
        "identity": {"roster_source": "sol_execbench", "slug": "fixture"},
        "invocation": {"adapter_kind": "sol_execbench"},
    }
    assert cc.validate_capture_cell(
        cell, cell_budgets=object(), workload_row_binding=binding,
        supplemental_runnability_evidence=evidence,
        runnability_path=retained,
    ) is sentinel


def _label_driver_ncu_identity_args(tmp_path: Path) -> list[str]:
    ncu = tmp_path / "ncu"
    ncu.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == --version ]]; then\n"
        "  printf 'Version 2026.1.0.0 (public-release)\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 99\n"
    )
    ncu.chmod(0o700)
    return [
        "--ncu", str(ncu),
        "--ncu-executable-sha256", hashlib.sha256(ncu.read_bytes()).hexdigest(),
        "--ncu-version-line", "Version 2026.1.0.0 (public-release)",
    ]


def test_label_driver_logical_binding_accepts_adapter_argv_before_ncu(tmp_path):
    missing_manifest = tmp_path / "missing.json"
    result = subprocess.run([
        str(cc.LABEL_DRIVER),
        "--device", "sm_86", "--slug", "sol_bridge_test",
        "--shape-point", "workloads",
        "--shape-binding", "logical-workload",
        "--repeat", "0", "--launch", "0",
        "--binary", "/bin/true",
        *_label_driver_ncu_identity_args(tmp_path),
        "--out-dir", str(tmp_path / "labels"),
        "--measurement-manifest", str(missing_manifest),
        "--", "benchmark_dir", "--solution", "solution.json", "--json",
    ], cwd=cc.REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "--measurement-manifest path does not exist" in result.stderr
    assert "do not unambiguously bind" not in result.stderr


def test_label_driver_logical_binding_refuses_fake_shape_token(tmp_path):
    result = subprocess.run([
        str(cc.LABEL_DRIVER),
        "--device", "sm_86", "--slug", "sol_bridge_test",
        "--shape-point", "workloads",
        "--shape-binding", "logical-workload",
        "--repeat", "0", "--launch", "0",
        "--binary", "/bin/true",
        *_label_driver_ncu_identity_args(tmp_path),
        "--out-dir", str(tmp_path / "labels"),
        "--", "benchmark_dir", "--shape-point", "workloads",
    ], cwd=cc.REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "logical-workload binding requires" in result.stderr


def test_live_guard_refuses_process_or_clock(monkeypatch):
    monkeypatch.setattr(
        cc.preflight, "probe_capture_processes", lambda: {"status": "failed"}
    )
    monkeypatch.setattr(
        cc.preflight, "probe_clock_state", lambda _: {"status": "ok"}
    )
    with pytest.raises(cc.ChunkRefusal, match="capture_process_conflict"):
        cc.live_capture_guard({})


def test_live_guard_waits_for_exact_clock_and_rechecks_process(monkeypatch):
    process_calls = []

    def processes():
        process_calls.append(len(process_calls) + 1)
        return {"status": "ok", "snapshot": len(process_calls)}

    monkeypatch.setattr(cc.preflight, "probe_capture_processes", processes)
    monkeypatch.setattr(
        cc.preflight, "wait_for_clock_state",
        lambda _: {"status": "ok", "settle_attempt_count": 2},
    )
    result = cc.live_capture_guard({})
    assert process_calls == [1, 2]
    assert result["capture_processes"]["snapshot"] == 2
    assert result["capture_processes_before_clock_wait"]["snapshot"] == 1
    assert result["clock_state"]["settle_attempt_count"] == 2


def test_live_guard_refuses_unsettled_clock(monkeypatch):
    monkeypatch.setattr(
        cc.preflight, "probe_capture_processes", lambda: {"status": "ok"}
    )
    monkeypatch.setattr(
        cc.preflight, "wait_for_clock_state",
        lambda _: {"status": "mismatch", "detail": "still P2"},
    )
    with pytest.raises(cc.ChunkRefusal, match="clock_state_drift"):
        cc.live_capture_guard({})


def test_run_logged_translates_process_start_failure(tmp_path, monkeypatch):
    def fail_start(*_args, **_kwargs):
        raise FileNotFoundError("synthetic executable missing")

    monkeypatch.setattr(cc.subprocess, "Popen", fail_start)
    with pytest.raises(cc.ChunkRefusal, match="probe_start_failed"):
        cc._run_logged(
            ["/missing/synthetic"], cwd=tmp_path, env=None, timeout_s=1,
            log_dir=tmp_path / "logs", stem="start", failure_code="probe",
        )
    assert (tmp_path / "logs/start.stdout.log").read_bytes() == b""
    assert "FileNotFoundError" in (
        tmp_path / "logs/start.stderr.log"
    ).read_text()


def test_run_logged_routes_supervised_ncu_and_writes_logs(
    tmp_path, monkeypatch,
):
    expected = _expected_ncu_identity()
    environment = cc.ncu_executor.NcuCollectionEnvironment({
        "LANG": "C", "LANGUAGE": "C", "LC_ALL": "C",
        "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
        "PWD": str(tmp_path), "PYTHONNOUSERSITE": "1",
        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
    })
    command = [
        "/bin/true", cc.ncu_executor.AUTHORIZED_NCU_PROC_PATH,
    ]
    observed = {}

    def execute(identity, action, **kwargs):
        observed.update(identity=identity, action=action, **kwargs)
        return cc.ncu_executor.NcuActionResult(0, b"stdout", b"stderr")

    monkeypatch.setattr(
        cc.ncu_executor, "execute_ncu_supervised_child", execute,
    )
    elapsed, stdout, stderr = cc._run_logged(
        command, cwd=tmp_path, env=None, timeout_s=5,
        log_dir=tmp_path / "logs", stem="supervised", failure_code="label",
        supervised_ncu_identity=expected,
        supervised_ncu_environment=environment,
    )
    assert elapsed >= 0
    assert stdout.read_bytes() == b"stdout"
    assert stderr.read_bytes() == b"stderr"
    assert observed == {
        "identity": expected,
        "action": command,
        "timeout_seconds": 5,
        "environment": environment,
        "working_directory": tmp_path,
    }


def test_run_logged_types_supervised_ncu_timeout(tmp_path, monkeypatch):
    expected = _expected_ncu_identity()
    environment = cc.ncu_executor.NcuCollectionEnvironment({
        "LANG": "C", "LANGUAGE": "C", "LC_ALL": "C",
        "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
        "PWD": str(tmp_path), "PYTHONNOUSERSITE": "1",
        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
    })

    def timeout(*_args, **_kwargs):
        error = cc.ncu_executor.NcuActionExecutorError(
            cc.ncu_executor.NCU_ACTION_TIMEOUT, "synthetic timeout",
        )
        error.stdout = b"partial stdout"
        error.stderr = b"partial stderr"
        raise error

    monkeypatch.setattr(
        cc.ncu_executor, "execute_ncu_supervised_child", timeout,
    )
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc._run_logged(
            ["/bin/true", cc.ncu_executor.AUTHORIZED_NCU_PROC_PATH],
            cwd=tmp_path, env=None, timeout_s=5,
            log_dir=tmp_path / "logs", stem="timeout", failure_code="label",
            supervised_ncu_identity=expected,
            supervised_ncu_environment=environment,
        )
    assert caught.value.code == "label_timeout"
    assert (tmp_path / "logs/timeout.stdout.log").read_bytes() == b"partial stdout"
    stderr = (tmp_path / "logs/timeout.stderr.log").read_text()
    assert "partial stderr" in stderr
    assert "NCU_ACTION_TIMEOUT" in stderr


def test_run_logged_storage_marker_precedes_supervised_timeout(
    tmp_path, monkeypatch,
):
    expected = _expected_ncu_identity()
    environment = cc.ncu_executor.NcuCollectionEnvironment({
        "LANG": "C", "LANGUAGE": "C", "LC_ALL": "C",
        "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
        "PWD": str(tmp_path), "PYTHONNOUSERSITE": "1",
        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
    })
    marker = cc.LABEL_DRIVER_STORAGE_MARKER + json.dumps({
        "schema_version": 1,
        "errno": errno.ENOSPC,
        "operation": "label_driver_output_probe",
    }, sort_keys=True, separators=(",", ":"))

    def timeout(*_args, **_kwargs):
        error = cc.ncu_executor.NcuActionExecutorError(
            cc.ncu_executor.NCU_ACTION_TIMEOUT, "synthetic timeout",
        )
        error.stdout = b"partial stdout"
        error.stderr = (marker + "\n").encode()
        raise error

    monkeypatch.setattr(
        cc.ncu_executor, "execute_ncu_supervised_child", timeout,
    )
    with pytest.raises(cc.ChunkRefusal) as caught:
        cc._run_logged(
            ["/bin/true", cc.ncu_executor.AUTHORIZED_NCU_PROC_PATH],
            cwd=tmp_path, env=None, timeout_s=5,
            log_dir=tmp_path / "logs", stem="storage", failure_code="label",
            trusted_child_storage_marker=True,
            supervised_ncu_identity=expected,
            supervised_ncu_environment=environment,
        )
    assert caught.value.code == "label_storage_failure"
    assert cc._storage_origin(caught.value).errno == errno.ENOSPC
    assert not (tmp_path / "logs/storage.stdout.log").exists()
    assert not (tmp_path / "logs/storage.stderr.log").exists()


def test_run_logged_kills_process_group_at_tree_limit(tmp_path):
    monitored = tmp_path / "trace"
    monitored.mkdir()
    command = [
        cc.sys.executable,
        "-c",
        (
            "from pathlib import Path; import time; "
            f"Path({str(monitored / 'payload.pb')!r}).write_bytes(b'x'*4096); "
            "time.sleep(30)"
        ),
    ]
    with pytest.raises(cc.ChunkRefusal, match="probe_tree_limit_exceeded"):
        cc._run_logged(
            command, cwd=tmp_path, env=None, timeout_s=5,
            log_dir=tmp_path / "logs", stem="limited", failure_code="probe",
            monitored_tree=monitored, tree_limit_bytes=512,
        )
    assert (monitored / "payload.pb").stat().st_size == 4096
    assert (tmp_path / "logs/limited.stdout.log").is_file()
    assert (tmp_path / "logs/limited.stderr.log").is_file()


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EIO, errno.EROFS])
def test_run_logged_storage_monitor_failure_never_writes_logs(
    tmp_path, monkeypatch, error_number,
):
    monitored = tmp_path / "trace"
    monitored.mkdir()
    logs = tmp_path / "logs"
    calls = 0

    def fail_monitor(_path):
        nonlocal calls
        calls += 1
        raise OSError(error_number, "injected monitored-tree storage fault")

    monkeypatch.setattr(cc, "_trace_tree_bytes", fail_monitor)
    command = [
        cc.sys.executable, "-c", "import time; time.sleep(30)",
    ]
    with pytest.raises(
        cc.ChunkRefusal, match="probe_storage_monitor_failed",
    ) as caught:
        cc._run_logged(
            command, cwd=tmp_path, env=None, timeout_s=5,
            log_dir=logs, stem="storage", failure_code="probe",
            monitored_tree=monitored, tree_limit_bytes=1024,
        )
    assert cc._storage_origin(caught.value).errno == error_number
    mapped = sol_chunk.storage_refusal_from_exception(
        caught.value, operation="capture_monitored_output_tree",
    )
    assert mapped is not None
    assert calls >= 1
    assert not (logs / "storage.stdout.log").exists()
    assert not (logs / "storage.stderr.log").exists()


def test_run_logged_final_storage_monitor_failure_never_writes_logs(
    tmp_path, monkeypatch,
):
    monitored = tmp_path / "trace"
    monitored.mkdir()
    logs = tmp_path / "logs"
    monkeypatch.setattr(
        cc, "_trace_tree_bytes",
        lambda _path: (_ for _ in ()).throw(
            OSError(errno.EIO, "injected final monitor read failure")
        ),
    )
    with pytest.raises(
        cc.ChunkRefusal, match="probe_storage_monitor_failed",
    ):
        cc._run_logged(
            [cc.sys.executable, "-c", "pass"],
            cwd=tmp_path, env=None, timeout_s=5,
            log_dir=logs, stem="final-storage", failure_code="probe",
            monitored_tree=monitored, tree_limit_bytes=1024,
        )
    assert not (logs / "final-storage.stdout.log").exists()
    assert not (logs / "final-storage.stderr.log").exists()


def test_run_logged_accepts_label_storage_marker_only_when_trusted(tmp_path):
    marker = (
        cc.LABEL_DRIVER_STORAGE_MARKER
        + json.dumps({
            "schema_version": 1,
            "errno": errno.EIO,
            "operation": "label_driver_output_probe",
        }, sort_keys=True, separators=(",", ":"))
    )
    command = [
        cc.sys.executable, "-c",
        f"import sys; sys.stderr.write({(marker + chr(10))!r}); sys.exit(1)",
    ]
    with pytest.raises(cc.ChunkRefusal) as spoofed:
        cc._run_logged(
            command, cwd=tmp_path, env=None, timeout_s=5,
            log_dir=tmp_path / "spoof-logs", stem="spoof",
            failure_code="spoofed_child",
        )
    assert spoofed.value.code == "spoofed_child"
    assert (tmp_path / "spoof-logs/spoof.stderr.log").is_file()

    with pytest.raises(
        cc.ChunkRefusal, match="trusted_child_storage_failure",
    ) as trusted:
        cc._run_logged(
            command, cwd=tmp_path, env=None, timeout_s=5,
            log_dir=tmp_path / "trusted-logs", stem="trusted",
            failure_code="trusted_child",
            trusted_child_storage_marker=True,
        )
    assert cc._storage_origin(trusted.value).errno == errno.EIO
    assert not (tmp_path / "trusted-logs/trusted.stdout.log").exists()
    assert not (tmp_path / "trusted-logs/trusted.stderr.log").exists()


def test_copy_output_first_storage_fault_wins_over_close_faults(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    class SourceStream:
        def __init__(self):
            self.read_once = False

        def read(self, _size):
            if not self.read_once:
                self.read_once = True
                return b"payload"
            return b""

        def close(self):
            raise OSError(errno.EIO, "secondary source close fault")

    class DestinationStream:
        def write(self, _payload):
            raise OSError(errno.ENOSPC, "primary destination write fault")

        def close(self):
            raise OSError(errno.EIO, "secondary destination close fault")

    def fake_open(path, *_args, **_kwargs):
        return SourceStream() if path == source else DestinationStream()

    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(cc.ChunkRefusal, match="copy_storage_failure") as caught:
        cc._copy_output_file(
            source, destination, exclusive=True,
            code="copy_storage_failure", operation="copy_fixture",
        )
    assert cc._storage_origin(caught.value).errno == errno.ENOSPC


def test_scoped_generated_output_reads_type_eio(tmp_path, monkeypatch):
    label = tmp_path / "label.json"
    label.write_text("{}\n")
    real_read = Path.read_bytes

    def fail_label(candidate):
        if candidate == label:
            raise OSError(errno.EIO, "injected label output read EIO")
        return real_read(candidate)

    monkeypatch.setattr(Path, "read_bytes", fail_label)
    with pytest.raises(
        cc.ChunkRefusal, match="label_capture_storage_failure",
    ):
        cc._json_output(
            label, code="label_capture_storage_failure",
            operation="label_fixture_read",
        )


@pytest.mark.parametrize(
    ("code", "operation"),
    [
        ("sol_control_storage_failure", "control_fixture_hash"),
        ("cell_inventory_storage_failure", "inventory_fixture_hash"),
    ],
)
def test_scoped_generated_output_hashes_type_eio(
    tmp_path, monkeypatch, code, operation,
):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"payload")
    real_open = Path.open

    def fail_open(path, *args, **kwargs):
        if path == artifact:
            raise OSError(errno.EIO, "injected output hash EIO")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(cc.ChunkRefusal, match=code):
        cc._sha256_output_file(artifact, code=code, operation=operation)


def test_campaign_chunk_lock_open_storage_failure_is_typed(tmp_path, monkeypatch):
    real_open = cc.os.open

    def fail_lock(path, *args, **kwargs):
        if Path(path).name == ".capture.lock":
            raise OSError(errno.ENOSPC, "injected lock open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(cc.os, "open", fail_lock)
    with pytest.raises(
        cc.ChunkRefusal, match="campaign_lock_storage_failure",
    ) as caught:
        cc.acquire_campaign_lock(tmp_path)
    assert cc._storage_origin(caught.value).errno == errno.ENOSPC


def test_campaign_chunk_lock_primary_storage_fault_wins_over_close_fault(
    tmp_path, monkeypatch,
):
    real_open = cc.os.open
    real_close = cc.os.close
    descriptor = None

    def track_lock_open(path, *args, **kwargs):
        nonlocal descriptor
        opened = real_open(path, *args, **kwargs)
        if Path(path).name == ".capture.lock":
            descriptor = opened
        return opened

    def fail_secondary_close(opened):
        if opened == descriptor:
            raise OSError(errno.EIO, "injected secondary lock close failure")
        return real_close(opened)

    real_fstat = cc.os.fstat

    def injected_fstat(opened):
        if opened == descriptor:
            raise OSError(errno.ENOSPC, "injected primary lock stat failure")
        return real_fstat(opened)

    monkeypatch.setattr(cc.os, "open", track_lock_open)
    monkeypatch.setattr(cc.os, "fstat", injected_fstat)
    monkeypatch.setattr(cc.os, "close", fail_secondary_close)
    try:
        with pytest.raises(
            cc.ChunkRefusal, match="campaign_lock_storage_failure",
        ) as caught:
            cc.acquire_campaign_lock(tmp_path)
        assert cc._storage_origin(caught.value).errno == errno.ENOSPC
    finally:
        if descriptor is not None:
            real_close(descriptor)


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_external_output_root_creation_storage_fault_is_typed(
    tmp_path, monkeypatch, error_number,
):
    parent = tmp_path / "external-parent"
    parent.mkdir()
    output = parent / "campaign"
    real_mkdir = Path.mkdir

    def injected_mkdir(path, *args, **kwargs):
        if path == output:
            raise OSError(error_number, "injected output-root mkdir failure")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", injected_mkdir)
    with pytest.raises(cc.ChunkRefusal, match="output_root_storage_failure") as caught:
        cc._assert_external_output_root(output)
    assert cc._storage_origin(caught.value).errno == error_number


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_journal_artifact_output_read_storage_fault_is_typed(
    tmp_path, monkeypatch, error_number,
):
    artifact = tmp_path / "cells" / "fixture" / "artifact.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture")
    result = {
        "artifacts": [{
            "relative_path": "cells/fixture/artifact.bin",
            "sha256": cc.sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }],
    }
    real_lstat = Path.lstat

    def injected_lstat(path, *args, **kwargs):
        if path == artifact:
            raise OSError(error_number, "injected journal artifact read failure")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", injected_lstat)
    with pytest.raises(cc.ChunkRefusal, match="cell_journal_storage_failure") as caught:
        cc._verify_journal_artifacts(
            result, root=tmp_path, safe_name="fixture",
        )
    assert cc._storage_origin(caught.value).errno == error_number


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_journal_artifact_output_resolve_storage_fault_is_typed(
    tmp_path, monkeypatch, error_number,
):
    artifact = tmp_path / "cells" / "fixture" / "artifact.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture")
    result = {
        "artifacts": [{
            "relative_path": "cells/fixture/artifact.bin",
            "sha256": cc.sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }],
    }
    real_resolve = Path.resolve

    def injected_resolve(path, *args, **kwargs):
        if path == artifact:
            raise OSError(error_number, "injected journal resolve failure")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", injected_resolve)
    with pytest.raises(
        cc.ChunkRefusal, match="cell_journal_storage_failure",
    ) as caught:
        cc._verify_journal_artifacts(
            result, root=tmp_path, safe_name="fixture",
        )
    assert cc._storage_origin(caught.value).errno == error_number


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_journal_artifact_output_traversal_storage_fault_is_typed(
    tmp_path, monkeypatch, error_number,
):
    artifact = tmp_path / "cells" / "fixture" / "artifact.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture")
    result = {
        "artifacts": [{
            "relative_path": "cells/fixture/artifact.bin",
            "sha256": cc.sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }],
    }
    real_scandir = cc.os.scandir

    def injected_scandir(path):
        if Path(path) == artifact.parent:
            raise OSError(error_number, "injected journal traversal failure")
        return real_scandir(path)

    monkeypatch.setattr(cc.os, "scandir", injected_scandir)
    with pytest.raises(
        cc.ChunkRefusal, match="cell_journal_storage_failure",
    ) as caught:
        cc._verify_journal_artifacts(
            result, root=tmp_path, safe_name="fixture",
        )
    assert cc._storage_origin(caught.value).errno == error_number


def test_label_driver_preserve_failed_evidence_contract_is_bound():
    script = cc.LABEL_DRIVER.read_text()
    assert script.count("LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE") == 4
    assert (
        '${LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE:-0}" != "1"'
        in script
    )


def test_output_root_must_be_absolute_and_external(tmp_path, monkeypatch):
    with pytest.raises(cc.ChunkRefusal, match="output_root_not_absolute"):
        cc._assert_external_output_root(Path("relative"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(cc, "REPO_ROOT", repo)
    inside = repo / ".test-capture-inside-repo"
    with pytest.raises(cc.ChunkRefusal, match="output_root_inside_repository"):
        cc._assert_external_output_root(inside)
    assert not inside.exists()


def test_resume_requires_exact_current_input_hashes(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"input-{index}"
        path.write_text(str(index))
        paths.append(path)
    fields = (
        "pins_sha256", "roster_authority_sha256", "trace_budget_sha256",
        "capture_matrix_sha256", "preflight_sha256",
    )
    manifest = {field: cc.sha256_file(path) for field, path in zip(fields, paths)}
    cc.validate_resume_inputs(
        manifest, pins_path=paths[0], authority_path=paths[1],
        aggregate_budget_path=paths[2], matrix_path=paths[3],
        preflight_path=paths[4],
    )
    paths[3].write_text("tampered")
    with pytest.raises(cc.ChunkRefusal, match="resume_input_mismatch"):
        cc.validate_resume_inputs(
            manifest, pins_path=paths[0], authority_path=paths[1],
            aggregate_budget_path=paths[2], matrix_path=paths[3],
            preflight_path=paths[4],
        )


def _write_completed_journal(
        stage: Path, cell: dict, *, preflight_sha: str = "c" * 64,
        source_commit: str = "d" * 40, matrix_sha: str = "b" * 64,
        matrix_seal: str = "e" * 64, run_id: str = "f" * 64,
        supersedes: dict | None = None,
        prior_preparation_failures: list[dict] | None = None,
        prior_capture_defect_attempts: list[dict] | None = None,
        complete: bool = True):
    safe = cc._cell_safe_name(cell)
    preflight_path = stage / f"preflight-{preflight_sha[:8]}.json"
    if not preflight_path.exists():
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text("{}\n")
    journal_dir = stage / "journals" / safe
    journal_dir.mkdir(parents=True)
    started = "2026-07-10T12:00:01Z"
    start = cc._journal_start_payload(
        cell=cell, matrix_sha256=matrix_sha,
        matrix_seal_sha256=matrix_seal,
        preflight_sha256=preflight_sha, preflight_path=preflight_path,
        source_commit=source_commit, capture_started_at_utc=started,
        admission_run_id=run_id, supersedes=supersedes,
        prior_preparation_failures=prior_preparation_failures or [],
        prior_capture_defect_attempts=prior_capture_defect_attempts or [],
    )
    cc._write_cell_journal(journal_dir / cc.JOURNAL_START_NAME, start)
    payload = journal_dir / "cells" / safe / "trace.pb"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"nonempty trace")
    cell_result_path = payload.parent / "cell_capture_result.json"
    cell_result = {
        "capture_preflight_sha256": preflight_sha,
        "capture_started_at_utc": started,
        "admission_run_id": run_id,
        "supersedes": supersedes,
        "prior_preparation_failures": prior_preparation_failures or [],
        "prior_capture_defect_attempts": prior_capture_defect_attempts or [],
        "definition_equivalence": None,
        "definition_equivalence_attestation_sha256": None,
    }
    cell_result_path.write_text(json.dumps(cell_result, sort_keys=True) + "\n")
    result = {
        "roster_source": cell["identity"]["roster_source"],
        "slug": cell["identity"]["slug"],
        "authority_cell_sha256": cell["authority_cell_sha256"],
        "disposition": "admitted",
        **cell_result,
        "artifacts": [
            {
                "artifact_kind": "trace_root",
                "relative_path": f"cells/{safe}/trace.pb",
                "sha256": cc.sha256_file(payload),
                "size_bytes": payload.stat().st_size,
            },
            {
                "artifact_kind": "cell_result",
                "relative_path": f"cells/{safe}/cell_capture_result.json",
                "sha256": cc.sha256_file(cell_result_path),
                "size_bytes": cell_result_path.stat().st_size,
            },
        ],
    }
    if complete:
        cc._write_cell_journal(
            journal_dir / cc.JOURNAL_COMPLETE_NAME,
            cc._journal_payload(start=start, result=result),
        )
    return journal_dir, start, result


def test_cell_journal_recovers_verified_attempt_without_recapture(tmp_path):
    cell = {
        "identity": {"roster_source": "portable_lcd", "slug": "cell_a"},
        "authority_cell_sha256": "a" * 64,
    }
    stage = tmp_path / "stage"
    journal_dir, _start, result = _write_completed_journal(stage, cell)
    recovered = cc._load_or_recover_cell_journal(
        journal_dir=journal_dir, cell=cell, matrix_sha256="b" * 64,
        matrix_seal_sha256="e" * 64, preflight_sha256="c" * 64,
        source_commit="d" * 40, current_invocation_admitted=True,
        original_preflight_status="fresh",
    )
    assert recovered == result
    assert (journal_dir / "cells" / cc._cell_safe_name(cell) / "trace.pb").read_bytes() == b"nonempty trace"


def test_problem070_capture_defect_backref_propagates_start_result_and_complete(
    tmp_path,
):
    cell = {
        "identity": {
            "roster_source": "sol_execbench",
            "slug": "problem_070_mamba2_fused_intra_chunk_diagonal_computation",
        },
        "authority_cell_sha256": "a" * 64,
    }
    entry_id = "f1bd571a-33f8-4596-b177-8f6af83bc679"
    root = tmp_path / "capture_defect_attempts"
    refs = [{
        "ledger_entry_id": entry_id,
        "ledger_path": str((root / "capture_defect_ledger.jsonl").resolve()),
        "archive_path": str((root / f"capture-defect-{entry_id}").resolve()),
        "original_head": "d" * 40,
        "reason_code": "CAPTURE_DRIVER_INTEGRITY_FALSE_POSITIVE_BEFORE_ROI",
    }]
    journal, start, result = _write_completed_journal(
        tmp_path / "stage", cell, prior_capture_defect_attempts=refs,
    )
    complete = json.loads((journal / cc.JOURNAL_COMPLETE_NAME).read_text())
    cell_result = json.loads(next(journal.rglob("cell_capture_result.json")).read_text())
    assert start["prior_capture_defect_attempts"] == refs
    assert result["prior_capture_defect_attempts"] == refs
    assert complete["prior_capture_defect_attempts"] == refs
    assert complete["result"]["prior_capture_defect_attempts"] == refs
    assert cell_result["prior_capture_defect_attempts"] == refs


@pytest.mark.parametrize("bad_rule", ["SOL_DEF_EQUIV_V1", None])
def test_sol_journal_recovery_refuses_non_v2_or_absent_rule_id(
    tmp_path, bad_rule
):
    cell = {
        "identity": {"roster_source": "sol_execbench", "slug": "problem_x"},
        "authority_cell_sha256": "a" * 64,
    }
    journal_dir = tmp_path / "journals" / cc._cell_safe_name(cell)
    journal_dir.mkdir(parents=True)
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}\n")
    start = cc._journal_start_payload(
        cell=cell,
        matrix_sha256="b" * 64,
        matrix_seal_sha256="e" * 64,
        preflight_sha256="c" * 64,
        preflight_path=preflight,
        source_commit="d" * 40,
        capture_started_at_utc="2026-07-10T12:00:01Z",
        admission_run_id="f" * 64,
        supersedes=None,
        prior_preparation_failures=[],
        prior_capture_defect_attempts=[],
    )
    if bad_rule is None:
        start.pop("definition_equivalence_rule_id")
    else:
        start["definition_equivalence_rule_id"] = bad_rule
    start.pop("journal_start_seal_sha256")
    start["journal_start_seal_sha256"] = cc.canonical_sha256(start)
    (journal_dir / cc.JOURNAL_START_NAME).write_text(
        json.dumps(start, sort_keys=True) + "\n"
    )

    with pytest.raises(
        cc.ChunkRefusal, match="SOL_ATTESTATION_RULE_MISMATCH"
    ):
        cc._read_cell_journal(
            journal_dir=journal_dir,
            cell=cell,
            matrix_sha256="b" * 64,
            matrix_seal_sha256="e" * 64,
            source_commit="d" * 40,
            allow_incomplete=True,
        )


def test_cross_preflight_recovery_is_refused_and_journal_is_unchanged(tmp_path):
    # Tombstone: the former cross-fresh-preflight recovery expectation was
    # inverted by the Round-28 interrupted-chunk provenance adjudication.
    cell = {
        "identity": {"roster_source": "simbench_bw", "slug": "cell_b"},
        "authority_cell_sha256": "a" * 64,
    }
    stage = tmp_path / "stage"
    journal_dir, _start, _result = _write_completed_journal(stage, cell)
    before = cc._content_inventory(journal_dir)
    with pytest.raises(cc.ChunkRefusal, match="JOURNAL_PREFLIGHT_MISMATCH"):
        cc._load_or_recover_cell_journal(
            journal_dir=journal_dir, cell=cell, matrix_sha256="b" * 64,
            matrix_seal_sha256="e" * 64, preflight_sha256="9" * 64,
            source_commit="d" * 40, current_invocation_admitted=True,
            original_preflight_status="fresh",
        )
    assert cc._content_inventory(journal_dir) == before


def test_expired_cross_preflight_journal_is_supersedable_not_mutated(tmp_path):
    cell = {
        "identity": {"roster_source": "simbench_bw", "slug": "cell_b"},
        "authority_cell_sha256": "a" * 64,
    }
    stage = tmp_path / "stage"
    journal_dir, _start, _result = _write_completed_journal(stage, cell)
    before = cc._content_inventory(journal_dir)
    with pytest.raises(cc.ChunkRefusal, match="JOURNAL_PREFLIGHT_SUPERSEDABLE"):
        cc._load_or_recover_cell_journal(
            journal_dir=journal_dir, cell=cell, matrix_sha256="b" * 64,
            matrix_seal_sha256="e" * 64, preflight_sha256="9" * 64,
            source_commit="d" * 40, current_invocation_admitted=True,
            original_preflight_status="supersedable",
        )
    assert cc._content_inventory(journal_dir) == before


def test_supersession_atomically_preserves_whole_chunk_and_ledgers_each_cell(
        tmp_path):
    stage = tmp_path / "stage"
    cells = [
        {
            "identity": {"roster_source": "portable_lcd", "slug": "a"},
            "authority_cell_sha256": "a" * 64,
        },
        {
            "identity": {"roster_source": "simbench_bw", "slug": "b"},
            "authority_cell_sha256": "b" * 64,
        },
    ]
    for cell in cells:
        _write_completed_journal(stage, cell)
    active = cc._active_journals(
        journals_root=stage / "journals", cells=cells,
        matrix_sha256="b" * 64, matrix_seal_sha256="e" * 64,
        source_commit="d" * 40,
    )
    before = {
        safe: cc._content_inventory(stage / "journals" / safe)
        for safe in active
    }
    backrefs = cc._supersede_active_journals(
        journals_root=stage / "journals",
        superseded_root=stage / "superseded", active=active,
        current_preflight_sha256="9" * 64,
        session_id="session", chunk_id="chunk",
        source_commit="d" * 40, matrix_seal_sha256="e" * 64,
        allow_overflow=False,
    )
    assert list((stage / "journals").iterdir()) == []
    assert set(backrefs) == set(active)
    ledger = cc._load_supersession_ledger(
        stage / "superseded" / cc.SUPERSESSION_LEDGER_NAME
    )
    assert len(ledger) == 2
    assert {row["cell_id"] for row in ledger} == set(active)
    assert {row["reason_code"] for row in ledger} == {
        "PREFLIGHT_EXPIRED_BEFORE_DONE"
    }
    assert len({Path(row["destination_path"]).parent for row in ledger}) == 1
    for row in ledger:
        destination = Path(row["destination_path"])
        assert cc._content_inventory(destination) == before[row["cell_id"]]
        inventory = destination / cc.SUPERSESSION_INVENTORY_NAME
        assert cc.sha256_file(inventory) == row["inventory_sha256"]
        assert backrefs[row["cell_id"]] == {
            "ledger_entry_id": row["ledger_entry_id"],
            "superseded_preflight_sha256": "c" * 64,
            "superseded_path": row["destination_path"],
        }


def _ledger_row(cell_id: str, index: int) -> dict:
    base = {
        "schema_version": 1,
        "timestamp_utc": f"2026-07-10T12:00:0{index}Z",
        "session_id": "session",
        "chunk_id": "chunk",
        "cell_id": cell_id,
        "original_preflight_sha256": "1" * 64,
        "current_preflight_sha256": "2" * 64,
        "source_path": f"/old/{index}",
        "destination_path": f"/superseded/{index}/{cell_id}",
        "reason_code": "PREFLIGHT_EXPIRED_BEFORE_DONE",
        "operator_flag": "--supersede-stale-journals",
        "source_commit": "d" * 40,
        "matrix_seal_sha256": "e" * 64,
        "inventory_sha256": "3" * 64,
    }
    return {**base, "ledger_entry_id": cc.canonical_sha256(base)}


def test_third_supersession_requires_distinct_escalation_flag(tmp_path):
    stage = tmp_path / "stage"
    cell = {
        "identity": {"roster_source": "portable_lcd", "slug": "a"},
        "authority_cell_sha256": "a" * 64,
    }
    journal_dir, _start, _result = _write_completed_journal(stage, cell)
    safe = cc._cell_safe_name(cell)
    superseded = stage / "superseded"
    for index in range(2):
        cc._append_jsonl_fsync(
            superseded / cc.SUPERSESSION_LEDGER_NAME,
            _ledger_row(safe, index),
        )
    active = cc._active_journals(
        journals_root=stage / "journals", cells=[cell],
        matrix_sha256="b" * 64, matrix_seal_sha256="e" * 64,
        source_commit="d" * 40,
    )
    before = cc._content_inventory(journal_dir)
    with pytest.raises(cc.ChunkRefusal, match="SUPERSESSION_LIMIT_EXCEEDED"):
        cc._supersede_active_journals(
            journals_root=stage / "journals", superseded_root=superseded,
            active=active, current_preflight_sha256="9" * 64,
            session_id="session", chunk_id="chunk",
            source_commit="d" * 40, matrix_seal_sha256="e" * 64,
            allow_overflow=False,
        )
    assert cc._content_inventory(journal_dir) == before
    refs = cc._supersede_active_journals(
        journals_root=stage / "journals", superseded_root=superseded,
        active=active, current_preflight_sha256="9" * 64,
        session_id="session", chunk_id="chunk",
        source_commit="d" * 40, matrix_seal_sha256="e" * 64,
        allow_overflow=True,
    )
    newest = next(
        row for row in cc._load_supersession_ledger(
            superseded / cc.SUPERSESSION_LEDGER_NAME
        ) if row["ledger_entry_id"] == refs[safe]["ledger_entry_id"]
    )
    assert "--allow-supersession-overflow" in newest["operator_flag"]


def _preparation_backref(index: int) -> dict:
    return {
        "ledger_entry_id": f"00000000-0000-4000-8000-{index:012d}",
        "ledger_path": "/campaign/preparation_failures/"
        "preparation_failure_ledger.jsonl",
        "archive_path": f"/campaign/preparation_failures/archive-{index}",
    }


def test_preparation_attempt_limit_is_separate_and_explicit():
    backrefs = [_preparation_backref(1), _preparation_backref(2)]
    with pytest.raises(cc.ChunkRefusal, match="PREPARATION_ATTEMPT_LIMIT"):
        cc._require_preparation_attempt_available(
            safe_name="sol_execbench__problem_070", backrefs=backrefs,
            allow_overflow=False,
        )
    cc._require_preparation_attempt_available(
        safe_name="sol_execbench__problem_070", backrefs=backrefs,
        allow_overflow=True,
    )


def test_active_journal_backrefs_must_equal_ledger_set():
    journal_refs = [_preparation_backref(1)]
    active = {
        "sol_execbench__problem_070": (
            {"prior_preparation_failures": journal_refs}, None,
        )
    }
    cc._require_active_preparation_backrefs(
        active, {"sol_execbench__problem_070": journal_refs},
    )
    with pytest.raises(
        cc.ChunkRefusal, match="PREPARATION_LEDGER_INCONSISTENT"
    ):
        cc._require_active_preparation_backrefs(
            active,
            {"sol_execbench__problem_070": [
                *journal_refs, _preparation_backref(2),
            ]},
        )


def test_capture_defect_gate_refuses_foreign_journal_before_cell_work(tmp_path):
    journals = (
        tmp_path / ".staging" / ("a" * 40) / "foreign-chunk"
        / ("c" * 64) / "journals" / "foreign-cell"
    )
    journals.mkdir(parents=True)
    with pytest.raises(
        cc.ChunkRefusal, match="CROSS_HEAD_ACTIVE_JOURNAL_UNRESOLVED",
    ):
        cc._capture_defect_state(
            output_root=tmp_path, cells=[], source_commit=HEAD,
        )


def test_supersession_refuses_cross_device_before_move(tmp_path, monkeypatch):
    cell = {
        "identity": {"roster_source": "portable_lcd", "slug": "a"},
        "authority_cell_sha256": "a" * 64,
    }
    stage = tmp_path / "stage"
    journal_dir, _start, _result = _write_completed_journal(stage, cell)
    active = cc._active_journals(
        journals_root=stage / "journals", cells=[cell],
        matrix_sha256="b" * 64, matrix_seal_sha256="e" * 64,
        source_commit="d" * 40,
    )
    superseded = stage / "superseded"
    real_stat = Path.stat

    def different_device(path, *args, **kwargs):
        observed = real_stat(path, *args, **kwargs)
        if path == superseded:
            fields = list(observed)
            fields[2] += 1
            return cc.os.stat_result(fields)
        return observed

    monkeypatch.setattr(Path, "stat", different_device)
    with pytest.raises(
        cc.ChunkRefusal, match="SUPERSESSION_CROSS_DEVICE_REFUSED"
    ):
        cc._supersede_active_journals(
            journals_root=stage / "journals", superseded_root=superseded,
            active=active, current_preflight_sha256="9" * 64,
            session_id="session", chunk_id="chunk",
            source_commit="d" * 40, matrix_seal_sha256="e" * 64,
            allow_overflow=False,
        )
    assert journal_dir.is_dir()


def test_unledgered_superseded_directory_refuses_before_new_attempt(tmp_path):
    orphan = tmp_path / "superseded" / "attempt" / "portable_lcd__cell"
    orphan.mkdir(parents=True)
    (orphan / "journal_start.json").write_text("{}\n")
    with pytest.raises(
        cc.ChunkRefusal, match="SUPERSESSION_LEDGER_INCONSISTENT"
    ):
        cc._validate_superseded_tree(tmp_path / "superseded")


class _SimulatedInterruption(RuntimeError):
    pass


def _runner_recovery_case(tmp_path: Path, monkeypatch):
    source_commit = "d" * 40
    cells = [
        {
            "identity": {"roster_source": "portable_lcd", "slug": "a"},
            "authority_cell_sha256": "a" * 64,
            "disposition": "runnable",
            "capture_readiness": {"status": "ready", "blocking_reasons": []},
        },
        {
            "identity": {"roster_source": "simbench_bw", "slug": "b"},
            "authority_cell_sha256": "b" * 64,
            "disposition": "runnable",
            "capture_readiness": {"status": "ready", "blocking_reasons": []},
        },
    ]
    members = [cell["identity"] for cell in cells]
    matrix = {
        "source_commit": source_commit,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "matrix_seal_sha256": "e" * 64,
        "input_pins": {
            "aggregate_retention_budget": {"raw_staging_margin_bytes": 1024},
            "sol_native_env_fingerprint": {
                "path": "fingerprint.json", "sha256": "1" * 64,
                "native_env_hash": "2" * 64,
            },
        },
        "cells": cells,
        "chunks": [{
            "chunk_id": "chunk",
            "membership_sha256": cc.canonical_sha256(members),
            "cells": members,
        }],
    }
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    matrix_path = inputs / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True) + "\n")
    pins_path = inputs / "pins.json"
    pins_path.write_text(json.dumps({
        "arch": "sm_86",
        "measurement_protocol": {"version": "v2.1"},
        "toolchain": {
            "ncu_version_prefix": "2026.1", "driver_version": "610",
            "nvcc_release": "13.3", "nvbit_version": "1.8",
        },
    }) + "\n")
    paths = {
        "matrix": matrix_path,
        "authority": inputs / "authority.json",
        "pins": pins_path,
        "aggregate_budget": inputs / "aggregate.json",
        "cell_budgets": inputs / "cells.toml",
        "metric_registry": inputs / "registry.toml",
    }
    for name, path in paths.items():
        if name not in {"matrix", "pins"}:
            path.write_text("{}\n")
    preflights = {}
    for name in ("A", "B"):
        path = inputs / f"preflight-{name}.json"
        path.write_text(json.dumps({"preflight": name}) + "\n")
        preflights[name] = path
    output_root = tmp_path / "output"
    probe = {
        "ambient": {}, "environment_contract": {},
        "environment_contract_sha256": "3" * 64,
        "constructed_environment_delta": {}, "matrix_pin": {},
        "matrix_sha256": cc.sha256_file(matrix_path),
        "observed_fingerprint": {}, "compiler_identity": {},
        "nvcc_release": {"major": 13, "minor": 2},
        "nvcc_version_output": "nvcc",
    }
    freshness = {path.resolve(): "fresh" for path in preflights.values()}
    validated: list[tuple[Path, bool]] = []
    generated_at = cc._utc_text(datetime.now(timezone.utc) - timedelta(seconds=1))

    def validate_referenced(*, preflight_path, expected_sha256, require_fresh,
                            **_kwargs):
        resolved = preflight_path.resolve()
        assert cc.sha256_file(resolved) == expected_sha256
        validated.append((resolved, require_fresh))
        if require_fresh and freshness[resolved] == "expired":
            raise cc.ChunkRefusal("PREFLIGHT_EXPIRED", str(resolved))
        return ({
            "generated_at_utc": generated_at,
            "checks": {
                "sol_environment": probe,
                "injection_runtime": {"tracer_so_sha256": "4" * 64},
            },
        }, _expected_ncu_identity())

    capture_counts = {"a": 0, "b": 0}

    def fake_capture_cell(
        *, cell, cell_dir, capture_provenance, expected_ncu_identity, **_kwargs,
    ):
        assert expected_ncu_identity == _expected_ncu_identity()
        slug = cell["identity"]["slug"]
        capture_counts[slug] += 1
        payload = cell_dir / "trace.pb"
        payload.write_bytes(f"capture-{slug}-{capture_counts[slug]}".encode())
        cell_result_path = cell_dir / "cell_capture_result.json"
        provenance = {
            "capture_preflight_sha256": capture_provenance["preflight_sha256"],
            "capture_started_at_utc": capture_provenance[
                "capture_started_at_utc"
            ],
            "admission_run_id": capture_provenance["admission_run_id"],
            "supersedes": capture_provenance["supersedes"],
            "prior_preparation_failures": capture_provenance[
                "prior_preparation_failures"
            ],
            "prior_capture_defect_attempts": capture_provenance[
                "prior_capture_defect_attempts"
            ],
            "definition_equivalence": None,
            "definition_equivalence_attestation_sha256": None,
        }
        cell_result_path.write_text(json.dumps(provenance, sort_keys=True) + "\n")

        def artifact(path, kind):
            return {
                "artifact_kind": kind,
                "relative_path": path.relative_to(cell_dir.parents[1]).as_posix(),
                "sha256": cc.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

        return {
            "roster_source": cell["identity"]["roster_source"],
            "slug": slug,
            "authority_cell_sha256": cell["authority_cell_sha256"],
            "disposition": "admitted", "typed_reason": None, "detail": None,
            **provenance,
            "artifacts": [
                artifact(payload, "trace_root"),
                artifact(cell_result_path, "cell_result"),
            ],
        }

    finalized: list[dict] = []

    def fake_finalize(**kwargs):
        hashes = {
            start["capture_preflight_sha256"]
            for start, _result in kwargs["active"].values()
        }
        assert len(hashes) == 1
        marker = {
            "preflight_sha256": next(iter(hashes)),
            "capture_counts": dict(capture_counts),
            "run_rows": kwargs["run_rows"],
        }
        finalized.append(marker)
        final_dir = kwargs["final_dir"]
        final_dir.mkdir(parents=True)
        (final_dir / "DONE.json").write_text(json.dumps(marker) + "\n")
        return final_dir

    monkeypatch.setattr(cc, "validate_matrix_inputs", lambda *_args, **_kw: {})
    monkeypatch.setattr(cc, "_git_state", lambda: (source_commit, True, ""))
    monkeypatch.setattr(cc, "_validate_referenced_preflight", validate_referenced)
    monkeypatch.setattr(
        cc.sol_env, "certify_sol_environment",
        lambda *_args, **_kw: (probe, dict(CERTIFYING_SOL_ENVIRONMENT)),
    )
    monkeypatch.setattr(
        cc.sol_env, "probe_sol_environment",
        lambda *_args, **_kw: {"status": "ok"},
    )
    monkeypatch.setattr(cc, "load_trace_budgets", lambda *_args: object())
    monkeypatch.setattr(cc, "capture_cell", fake_capture_cell)
    monkeypatch.setattr(cc, "live_capture_guard", lambda *_args: {})
    monkeypatch.setattr(cc, "_finalize_from_journals", fake_finalize)

    def args(preflight: Path, *, supersede=False, overflow=False):
        return SimpleNamespace(
            matrix=str(paths["matrix"]), authority=str(paths["authority"]),
            pins=str(paths["pins"]),
            aggregate_budget=str(paths["aggregate_budget"]),
            cell_budgets=str(paths["cell_budgets"]),
            metric_registry=str(paths["metric_registry"]),
            preflight=str(preflight), chunk_id="chunk",
            output_root=str(output_root),
            supersede_stale_journals=supersede,
            allow_supersession_overflow=overflow,
        )

    return SimpleNamespace(
        args=args, preflights=preflights, freshness=freshness,
        capture_counts=capture_counts, finalized=finalized,
        output_root=output_root, matrix_path=matrix_path,
        validated=validated,
    )


def _interrupt_after_first_complete(monkeypatch):
    original = cc._write_cell_journal
    interrupted = False

    def write(path, payload):
        nonlocal interrupted
        original(path, payload)
        if path.name == cc.JOURNAL_COMPLETE_NAME and not interrupted:
            interrupted = True
            raise _SimulatedInterruption("after first completed cell")

    monkeypatch.setattr(cc, "_write_cell_journal", write)
    return original


def test_fast_resume_same_preflight_recovers_without_recapturing_first_cell(
        tmp_path, monkeypatch):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    original_write = _interrupt_after_first_complete(monkeypatch)
    with pytest.raises(_SimulatedInterruption):
        cc.run_chunk(case.args(case.preflights["A"]))
    assert case.capture_counts == {"a": 1, "b": 0}
    monkeypatch.setattr(cc, "_write_cell_journal", original_write)
    cc.run_chunk(case.args(case.preflights["A"]))
    assert case.capture_counts == {"a": 1, "b": 1}
    assert case.finalized[-1]["preflight_sha256"] == cc.sha256_file(
        case.preflights["A"]
    )
    assert len([
        row for row in case.finalized[-1]["run_rows"]
        if row.get("event") == "capture_admission"
    ]) == 2


def test_cross_head_finalized_nonbootstrap_requires_revision_archive(
    tmp_path, monkeypatch,
):
    import campaign_done_manifest as done

    case = _runner_recovery_case(tmp_path, monkeypatch)
    final_dir = case.output_root / "chunks" / "chunk"
    final_dir.mkdir(parents=True)
    done_path = final_dir / "DONE.json"
    done_path.write_text(json.dumps({"historical": True}) + "\n")
    before = done_path.read_bytes()

    def verify(_manifest, **kwargs):
        return ["source HEAD mismatch"] if kwargs.get("require_current_head") else []

    monkeypatch.setattr(done, "verify_seal", verify)
    monkeypatch.setattr(cc, "_historical_chunk_matches", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(done, "_is_historical_bootstrap", lambda _manifest: False)
    with pytest.raises(
        cc.ChunkRefusal, match="historical non-bootstrap chunk must be revision-archived"
    ):
        cc.run_chunk(case.args(case.preflights["A"]))
    assert done_path.read_bytes() == before
    assert case.capture_counts == {"a": 0, "b": 0}


def test_cross_head_exact_historical_bootstrap_remains_resumable(
    tmp_path, monkeypatch,
):
    import campaign_done_manifest as done

    case = _runner_recovery_case(tmp_path, monkeypatch)
    final_dir = case.output_root / "chunks" / "chunk"
    final_dir.mkdir(parents=True)
    (final_dir / "DONE.json").write_text(json.dumps({"bootstrap": True}) + "\n")

    def verify(_manifest, **kwargs):
        return ["source HEAD mismatch"] if kwargs.get("require_current_head") else []

    monkeypatch.setattr(done, "verify_seal", verify)
    monkeypatch.setattr(cc, "_historical_chunk_matches", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(done, "_is_historical_bootstrap", lambda _manifest: True)
    assert cc.run_chunk(case.args(case.preflights["A"])) == final_dir
    assert case.capture_counts == {"a": 0, "b": 0}


def test_fresh_mismatched_preflight_refuses_even_with_supersession_flag(
        tmp_path, monkeypatch):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    original_write = _interrupt_after_first_complete(monkeypatch)
    with pytest.raises(_SimulatedInterruption):
        cc.run_chunk(case.args(case.preflights["A"]))
    monkeypatch.setattr(cc, "_write_cell_journal", original_write)
    stage = (
        case.output_root / ".staging" / ("d" * 40) / "chunk"
        / cc.sha256_file(case.matrix_path)
    )
    before = cc._content_inventory(stage / "journals" / "portable_lcd__a")
    with pytest.raises(cc.ChunkRefusal, match="JOURNAL_PREFLIGHT_MISMATCH"):
        cc.run_chunk(case.args(case.preflights["B"], supersede=True))
    assert cc._content_inventory(
        stage / "journals" / "portable_lcd__a"
    ) == before


def test_expired_mismatch_without_flag_refuses_without_mutation(
        tmp_path, monkeypatch):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    original_write = _interrupt_after_first_complete(monkeypatch)
    with pytest.raises(_SimulatedInterruption):
        cc.run_chunk(case.args(case.preflights["A"]))
    monkeypatch.setattr(cc, "_write_cell_journal", original_write)
    case.freshness[case.preflights["A"].resolve()] = "expired"
    stage = (
        case.output_root / ".staging" / ("d" * 40) / "chunk"
        / cc.sha256_file(case.matrix_path)
    )
    before = cc._content_inventory(stage / "journals" / "portable_lcd__a")
    with pytest.raises(
        cc.ChunkRefusal, match="JOURNAL_PREFLIGHT_SUPERSEDABLE"
    ):
        cc.run_chunk(case.args(case.preflights["B"]))
    assert cc._content_inventory(
        stage / "journals" / "portable_lcd__a"
    ) == before


def test_expired_mismatch_with_flag_supersedes_and_recaptures_entire_chunk(
        tmp_path, monkeypatch):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    original_write = _interrupt_after_first_complete(monkeypatch)
    with pytest.raises(_SimulatedInterruption):
        cc.run_chunk(case.args(case.preflights["A"]))
    monkeypatch.setattr(cc, "_write_cell_journal", original_write)
    case.freshness[case.preflights["A"].resolve()] = "expired"
    cc.run_chunk(case.args(case.preflights["B"], supersede=True))
    assert case.capture_counts == {"a": 2, "b": 1}
    assert case.finalized[-1]["preflight_sha256"] == cc.sha256_file(
        case.preflights["B"]
    )
    stage = (
        case.output_root / ".staging" / ("d" * 40) / "chunk"
        / cc.sha256_file(case.matrix_path)
    )
    ledger = cc._validate_superseded_tree(stage / "superseded")
    assert len(ledger) == 1
    active = cc._active_journals(
        journals_root=stage / "journals",
        cells=json.loads(case.matrix_path.read_text())["cells"],
        matrix_sha256=cc.sha256_file(case.matrix_path),
        matrix_seal_sha256="e" * 64, source_commit="d" * 40,
    )
    assert active["portable_lcd__a"][1]["supersedes"][
        "ledger_entry_id"
    ] == ledger[0]["ledger_entry_id"]


def test_finalization_only_ignores_expiry_and_stray_fresh_preflight(
        tmp_path, monkeypatch):
    case = _runner_recovery_case(tmp_path, monkeypatch)

    def interrupt_finalization(**_kwargs):
        raise _SimulatedInterruption("after all cells, before DONE")

    monkeypatch.setattr(cc, "_finalize_from_journals", interrupt_finalization)
    with pytest.raises(_SimulatedInterruption):
        cc.run_chunk(case.args(case.preflights["A"]))
    assert case.capture_counts == {"a": 1, "b": 1}
    case.freshness[case.preflights["A"].resolve()] = "expired"
    validations_before = list(case.validated)

    def finish(**kwargs):
        hashes = {
            start["capture_preflight_sha256"]
            for start, _result in kwargs["active"].values()
        }
        case.finalized.append({"preflight_sha256": next(iter(hashes))})
        kwargs["final_dir"].mkdir(parents=True)
        (kwargs["final_dir"] / "DONE.json").write_text("{}\n")
        return kwargs["final_dir"]

    monkeypatch.setattr(cc, "_finalize_from_journals", finish)
    cc.run_chunk(case.args(case.preflights["B"]))
    assert case.finalized[-1]["preflight_sha256"] == cc.sha256_file(
        case.preflights["A"]
    )
    assert case.validated == validations_before


def test_incomplete_journal_is_capture_remaining_and_expiry_is_enforced(
        tmp_path, monkeypatch):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    original_capture = cc.capture_cell
    monkeypatch.setattr(
        cc, "capture_cell",
        lambda **_kwargs: (_ for _ in ()).throw(
            _SimulatedInterruption("mid-cell")
        ),
    )
    with pytest.raises(_SimulatedInterruption):
        cc.run_chunk(case.args(case.preflights["A"]))
    monkeypatch.setattr(cc, "capture_cell", original_capture)
    case.freshness[case.preflights["A"].resolve()] = "expired"
    with pytest.raises(cc.ChunkRefusal, match="PREFLIGHT_EXPIRED"):
        cc.run_chunk(case.args(case.preflights["A"]))


def test_chunk_gate_refuses_ambient_tripwire_before_capture(tmp_path, monkeypatch):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cc.sol_env, "certify_sol_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cc.sol_env.SolEnvironmentRefusal(
                "SOL_ENV_AMBIENT_CONFLICT", "NVCC_CCBIN=/usr/bin/g++-15"
            )
        ),
    )
    with pytest.raises(cc.ChunkRefusal, match="SOL_ENV_AMBIENT_CONFLICT"):
        cc.run_chunk(case.args(case.preflights["A"]))
    assert case.capture_counts == {"a": 0, "b": 0}


def test_chunk_gate_checks_every_preparation_attempt_limit_before_first_capture(
    tmp_path, monkeypatch
):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cc,
        "_preparation_failure_backrefs_by_cell",
        lambda **_kwargs: {
            "portable_lcd__a": [],
            "simbench_bw__b": [{"attempt": 1}, {"attempt": 2}],
        },
    )

    with pytest.raises(cc.ChunkRefusal, match="PREPARATION_ATTEMPT_LIMIT"):
        cc.run_chunk(case.args(case.preflights["A"]))

    assert case.capture_counts == {"a": 0, "b": 0}


def test_runner_preparation_ledger_gate_is_strict_and_propagates_before_admission(
    tmp_path, monkeypatch
):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    unledgered_archive = (
        case.output_root.resolve()
        / cc.prep_archive.PREPARATION_FAILURE_ROOT_NAME
        / "20260711T120000Z__head-deadbeef__pf-feedface__chunk"
    )
    detail = f"archive evidence exists without ledger: {[str(unledgered_archive)]!r}"
    calls = []

    def refuse_unledgered_archive(path, *, verify_archives=False):
        calls.append((path, verify_archives))
        assert path == (
            case.output_root.resolve()
            / cc.prep_archive.PREPARATION_FAILURE_ROOT_NAME
            / cc.prep_archive.PREPARATION_FAILURE_LEDGER_NAME
        )
        assert verify_archives is True
        raise cc.prep_archive.PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", detail,
        )

    monkeypatch.setattr(
        cc.prep_archive,
        "load_preparation_failure_ledger",
        refuse_unledgered_archive,
    )

    with pytest.raises(cc.ChunkRefusal) as caught:
        cc.run_chunk(case.args(case.preflights["A"]))

    assert calls == [(
        case.output_root.resolve()
        / cc.prep_archive.PREPARATION_FAILURE_ROOT_NAME
        / cc.prep_archive.PREPARATION_FAILURE_LEDGER_NAME,
        True,
    )]
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    assert caught.value.detail == detail
    assert str(caught.value) == f"PREPARATION_LEDGER_INCONSISTENT: {detail}"
    assert case.validated == []
    assert case.capture_counts == {"a": 0, "b": 0}
    run_journals = list(case.output_root.rglob("chunk_run_journal.jsonl"))
    assert len(run_journals) == 1
    rows = [json.loads(line) for line in run_journals[0].read_text().splitlines()]
    assert not any(row.get("event") == "capture_admission" for row in rows)


def test_preparation_ledger_accepts_ancestor_replacement_head_without_rewrite(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    ledger_path = (
        output_root
        / cc.prep_archive.PREPARATION_FAILURE_ROOT_NAME
        / cc.prep_archive.PREPARATION_FAILURE_LEDGER_NAME
    )
    replacement_head = "a" * 40
    source_commit = "b" * 40
    entry_id = "d4cc26f0-60b7-4d26-9d17-b0a40c48b3ed"
    row = {
        "entry_id": entry_id,
        "cell_id": "portable_lcd__a",
        "replacement_head": replacement_head,
        "archive_path": str(tmp_path / "preserved-attempt"),
    }
    monkeypatch.setattr(
        cc.prep_archive,
        "load_preparation_failure_ledger",
        lambda path, *, verify_archives: (
            [row]
            if path == ledger_path.resolve() and verify_archives is True
            else pytest.fail("unexpected ledger load")
        ),
    )
    observed_commands = []

    def ancestor(command, **kwargs):
        observed_commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cc.subprocess, "run", ancestor)
    cells = [{
        "identity": {"roster_source": "portable_lcd", "slug": "a"},
    }]

    backrefs = cc._preparation_failure_backrefs_by_cell(
        output_root=output_root,
        cells=cells,
        source_commit=source_commit,
    )

    assert row["replacement_head"] == replacement_head
    assert observed_commands == [(
        [
            "git", "merge-base", "--is-ancestor",
            replacement_head, source_commit,
        ],
        {
            "cwd": cc.REPO_ROOT,
            "capture_output": True,
            "text": True,
            "check": False,
        },
    )]
    assert backrefs == {"portable_lcd__a": [{
        "ledger_entry_id": entry_id,
        "ledger_path": str(ledger_path.resolve()),
        "archive_path": row["archive_path"],
    }]}


def test_preparation_ledger_rejects_divergent_replacement_head(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    replacement_head = "a" * 40
    source_commit = "b" * 40
    row = {
        "entry_id": "d4cc26f0-60b7-4d26-9d17-b0a40c48b3ed",
        "cell_id": "portable_lcd__a",
        "replacement_head": replacement_head,
        "archive_path": str(tmp_path / "preserved-attempt"),
    }
    monkeypatch.setattr(
        cc.prep_archive,
        "load_preparation_failure_ledger",
        lambda *_args, **_kwargs: [row],
    )
    monkeypatch.setattr(
        cc.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", ""
        ),
    )

    with pytest.raises(cc.ChunkRefusal) as caught:
        cc._preparation_failure_backrefs_by_cell(
            output_root=output_root,
            cells=[{
                "identity": {"roster_source": "portable_lcd", "slug": "a"},
            }],
            source_commit=source_commit,
        )

    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    assert "is not an ancestor" in caught.value.detail


def test_ambient_tripwire_reappearance_on_resume_refuses_before_recovery(
        tmp_path, monkeypatch):
    case = _runner_recovery_case(tmp_path, monkeypatch)
    original_write = _interrupt_after_first_complete(monkeypatch)
    with pytest.raises(_SimulatedInterruption):
        cc.run_chunk(case.args(case.preflights["A"]))
    monkeypatch.setattr(cc, "_write_cell_journal", original_write)
    monkeypatch.setattr(
        cc.sol_env, "certify_sol_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cc.sol_env.SolEnvironmentRefusal(
                "SOL_ENV_AMBIENT_CONFLICT", "NVCC_CCBIN=/usr/bin/g++-15"
            )
        ),
    )
    with pytest.raises(cc.ChunkRefusal, match="SOL_ENV_AMBIENT_CONFLICT"):
        cc.run_chunk(case.args(case.preflights["A"]))
    assert case.capture_counts == {"a": 1, "b": 0}


def test_done_finalizer_derives_uniform_journal_preflight_not_cli_preflight(
        tmp_path, monkeypatch):
    import campaign_done_manifest as done

    stage = tmp_path / "stage"
    source_commit = "d" * 40
    cell = {
        "identity": {"roster_source": "portable_lcd", "slug": "a"},
        "authority_cell_sha256": "a" * 64,
    }
    journal_dir, start, _result = _write_completed_journal(
        stage, cell, source_commit=source_commit,
        matrix_sha="b" * 64, matrix_seal="e" * 64,
        run_id="f" * 64,
    )
    original_preflight = Path(start["capture_preflight_path"])
    active = cc._active_journals(
        journals_root=stage / "journals", cells=[cell],
        matrix_sha256="b" * 64, matrix_seal_sha256="e" * 64,
        source_commit=source_commit,
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = {}
    for name in ("matrix", "authority", "pins", "budget"):
        path = inputs / f"{name}.json"
        path.write_text("{}\n")
        paths[name] = path
    matrix = {
        "matrix_seal_sha256": "e" * 64,
        "arch": "sm_86",
    }
    pins = {
        "arch": "sm_86",
        "measurement_protocol": {"version": "v2.1"},
    }
    admission = {
        "schema_version": 1, "event": "capture_admission",
        "session_id": "session", "chunk_id": "chunk",
        "run_id": "f" * 64,
        "admitted_at_utc": "2026-07-10T12:00:00Z",
        "preflight_path": str(original_preflight),
        "preflight_sha256": "c" * 64,
        "preflight_generated_at_utc": "2026-07-10T11:59:59Z",
    }
    stray = {
        **admission,
        "run_id": "9" * 64,
        "preflight_sha256": "8" * 64,
        "preflight_path": str(tmp_path / "stray-B.json"),
    }
    captured = {}

    def build_done_manifest(**kwargs):
        captured.update(kwargs)
        return {
            "preflight_sha256": kwargs["preflight_sha256"],
            "input_paths": kwargs["input_paths"],
            "admission": kwargs["admission"],
        }

    monkeypatch.setattr(
        cc, "_validate_referenced_preflight",
        lambda **_kwargs: (
            {"generated_at_utc": "2026-07-10T11:59:59Z"},
            _expected_ncu_identity(),
        ),
    )
    monkeypatch.setattr(cc, "_git_state", lambda: (source_commit, True, ""))
    monkeypatch.setattr(done, "_tool_versions_from_pins", lambda _pins: {})
    monkeypatch.setattr(done, "build_done_manifest", build_done_manifest)
    monkeypatch.setattr(done, "_assert_schema_valid", lambda _manifest: None)
    monkeypatch.setattr(done, "verify_seal", lambda *_args, **_kwargs: [])
    final_dir = tmp_path / "output" / "chunks" / "chunk"
    result = cc._finalize_from_journals(
        args=SimpleNamespace(chunk_id="chunk"), cells=[cell], active=active,
        stage_root=stage, final_dir=final_dir, matrix=matrix,
        matrix_path=paths["matrix"], matrix_sha256="b" * 64,
        authority_path=paths["authority"], pins=pins,
        pins_path=paths["pins"], aggregate_budget_path=paths["budget"],
        output_root=tmp_path / "output", source_commit=source_commit,
        run_journal_path=stage / "chunk_run_journal.jsonl",
        run_rows=[admission, stray],
    )
    assert result == final_dir
    assert captured["preflight_sha256"] == "c" * 64
    assert captured["input_paths"]["preflight"] == str(original_preflight)
    assert captured["admission"]["capture_invocations"] == [{
        "run_id": "f" * 64,
        "admitted_at_utc": "2026-07-10T12:00:00Z",
    }]
    assert json.loads((final_dir / "DONE.json").read_text())[
        "preflight_sha256"
    ] == "c" * 64


def test_historical_chunk_must_match_current_authority_exactly():
    cells = [{
        "identity": {"roster_source": "portable_lcd", "slug": "a"},
        "authority_cell_sha256": "1" * 64,
    }]
    manifest = {
        "chunk_id": "chunk-a",
        "cell_results": [{
            "roster_source": "portable_lcd", "slug": "a",
            "authority_cell_sha256": "1" * 64, "disposition": "admitted",
        }],
    }
    assert cc._historical_chunk_matches(
        manifest, cells=cells, chunk_id="chunk-a"
    ) == []
    manifest["cell_results"][0]["authority_cell_sha256"] = "2" * 64
    assert any("authority hash drift" in problem for problem in
               cc._historical_chunk_matches(
                   manifest, cells=cells, chunk_id="chunk-a"))


@pytest.mark.campaign_evidence
def test_consumer_rederives_external_matrix_and_rejects_resealed_invocation_tamper(
        tmp_path):
    from tools.local_labels import build_sm86_capture_matrix as matrix_builder
    external_authority = tmp_path / "authority.json"
    external_authority.write_bytes(cc.DEFAULT_AUTHORITY.read_bytes())
    with _frozen_canonical_loader_context():
        matrix = matrix_builder.build_matrix(
            cc.REPO_ROOT, require_head_match=False,
            authority_path=external_authority,
        )
    external_matrix = tmp_path / "matrix.json"
    external_matrix.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    with _frozen_canonical_loader_context():
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                cc, "rederive_capture_matrix", matrix_builder.validate_matrix,
            )
            cc.validate_matrix_inputs(
                matrix, matrix_path=external_matrix,
                authority_path=external_authority,
                pins_path=cc.DEFAULT_PINS,
                aggregate_budget_path=cc.DEFAULT_AGGREGATE_BUDGET,
                cell_budget_path=cc.DEFAULT_CELL_BUDGETS,
                registry_path=cc.DEFAULT_REGISTRY,
            )
    tampered = json.loads(json.dumps(matrix))
    bootstrap = next(
        row for row in tampered["cells"]
        if row["identity"] == {
            "roster_source": "portable_lcd", "slug": "silu_fwd_bf16_sm80",
        }
    )
    bootstrap["invocation"]["trace_app_args"][-1] = "2"
    bootstrap["authority_cell_sha256"] = matrix_builder.canonical_sha256(
        next(
            row for row in json.loads(external_authority.read_text())["cells"]
            if row["roster_source"] == "portable_lcd"
            and row["slug"] == "silu_fwd_bf16_sm80"
        )
    )
    tampered["matrix_seal_sha256"] = matrix_builder.canonical_sha256({
        key: value for key, value in tampered.items()
        if key != "matrix_seal_sha256"
    })
    with _frozen_canonical_loader_context():
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                cc, "rederive_capture_matrix", matrix_builder.validate_matrix,
            )
            with pytest.raises(
                cc.ChunkRefusal, match="matrix_semantic_rederivation_failed"
            ):
                cc.validate_matrix_inputs(
                    tampered, matrix_path=external_matrix,
                    authority_path=external_authority, pins_path=cc.DEFAULT_PINS,
                    aggregate_budget_path=cc.DEFAULT_AGGREGATE_BUDGET,
                    cell_budget_path=cc.DEFAULT_CELL_BUDGETS,
                    registry_path=cc.DEFAULT_REGISTRY,
                )


def test_label_capture_stages_workload_output_and_checks_full_vector(
        tmp_path, monkeypatch):
    matrix = json.loads(cc.DEFAULT_MATRIX.read_text())
    cell = next(
        row for row in matrix["cells"]
        if row["identity"] == {
            "roster_source": "portable_lcd", "slug": "silu_fwd_bf16_sm80",
        }
    )
    pins = json.loads(cc.DEFAULT_PINS.read_text())
    binary = cc.REPO_ROOT / cell["invocation"]["executable_path"]
    source_label = (
        cc.REPO_ROOT / "data" / "annex" / "repeatability_matrix_v21_ref_20260706"
        / "cfg=clock_none__cache_all" / "labels" / "sm_86"
        / "silu_fwd_bf16_sm80" / "timing_shape" / "label_0_0.json"
    )
    cell_dir = tmp_path / "stage" / "cells" / "portable__silu"
    cell_dir.mkdir(parents=True)
    commands = []

    def fake_run(
        command, *, cwd, env, timeout_s, log_dir, stem, failure_code,
        trusted_child_storage_marker=False, supervised_ncu_identity=None,
        supervised_ncu_environment=None,
    ):
        assert supervised_ncu_identity == _expected_ncu_identity()
        assert isinstance(
            supervised_ncu_environment, cc.ncu_executor.NcuCollectionEnvironment,
        )
        commands.append(command)
        out = cell_dir / "labels" / "sm_86" / "silu_fwd_bf16_sm80" / "timing_shape"
        out.mkdir(parents=True)
        label = json.loads(source_label.read_text())
        label.update({
            "device": "sm_86", "slug": "silu_fwd_bf16_sm80",
            "shape_point": "timing_shape",
            "binary_sha256": cell["invocation"]["executable_sha256"],
            "profiler_version": "Version 2026.1.0.0 (public-release)",
            "cuda_version": "13.3", "driver_version": "610.43.02",
            "measurement_protocol": {
                "ncu_clock_control": "none", "ncu_cache_control": "all",
                "ncu_replay_mode": "kernel", "host_persistence_mode": "Enabled",
                "host_sm_clock_mhz": 1350, "host_mem_clock_mhz": 10501,
            },
        })
        label["launch_index"] = 1
        (out / "label_0_1.json").write_text(json.dumps(label) + "\n")
        (out / "label_run_manifest_0_1.json").write_text(json.dumps({
            "schema_version": 1, "source_commit": matrix["source_commit"],
            "binary_sha256": cell["invocation"]["executable_sha256"],
            "kernel_signature_sha256": "1" * 64,
            "kernel_signature_source": "fallback_legacy", "arch": "sm_86",
            "slug": "silu_fwd_bf16_sm80", "shape_point": "timing_shape",
            "repeat_index": 0, "launch_index": 1,
        }) + "\n")
        report = out / "run_0_1.ncu-rep"
        report.write_bytes(b"ncu report")
        _write_launch_observation(
            out / "run_0_1.launch_observation.json",
            report=report, launch_index=1,
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = log_dir / f"{stem}.stdout.log"
        stderr = log_dir / f"{stem}.stderr.log"
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        workload_index = command.index("--output_dir") + 1
        workload = Path(command[workload_index])
        workload.mkdir(parents=True, exist_ok=True)
        (workload / "manifest.json").write_text("{}\n")
        return 1.0, stdout, stderr

    monkeypatch.setattr(cc, "_run_logged", fake_run)
    result = cc.capture_label(
        cell=cell, binary=binary, invocation=cell["invocation"],
        cell_dir=cell_dir, matrix=matrix, pins=pins, timeout_s=10,
        expected_ncu_identity=_expected_ncu_identity(),
        kernel_name_selector="_Z15syntheticKernelv",
    )
    command = commands[0]
    assert command.count(cc.ncu_executor.AUTHORIZED_NCU_PROC_PATH) == 1
    assert command[command.index("--ncu-exec-path") + 1] is (
        cc.ncu_executor.AUTHORIZED_NCU_PROC_PATH
    )
    assert command[command.index("--launch") + 1] == "1"
    assert command[command.index("--kernel-name") + 1] == (
        "_Z15syntheticKernelv"
    )
    assert command[command.index("--warmup") + 1] == "1"
    output_arg = Path(command[command.index("--output_dir") + 1])
    assert output_arg.is_relative_to(cell_dir / "logs")
    assert result["unit_metric_count"] == 53
    assert result["capture_manifest"].is_file()


def test_sol_label_consumer_keeps_shape_metadata_out_of_target_argv(
        tmp_path, monkeypatch):
    matrix, cell = _generated_sol_cell()
    pins = json.loads(cc.DEFAULT_PINS.read_text())
    pins["toolchain"]["nvcc_release"] = "13.2"
    invocation = cell["invocation"]
    binary = cc.REPO_ROOT / invocation["executable_path"]
    shape = invocation["logical_workload"]["value"]
    slug = cell["identity"]["slug"]
    source_label = (
        cc.REPO_ROOT / "data" / "annex" / "repeatability_matrix_v21_ref_20260706"
        / "cfg=clock_none__cache_all" / "labels" / "sm_86"
        / "silu_fwd_bf16_sm80" / "timing_shape" / "label_0_0.json"
    )
    cell_dir = tmp_path / "stage" / "cells" / f"sol_execbench__{slug}"
    cell_dir.mkdir(parents=True)
    runtime_env = dict(cc.os.environ)
    runtime_env["CUTLASS_DIR"] = str(cc.REPO_ROOT / "third_party/cutlass")
    runtime_env.update({
        "LD_PRELOAD": "/hostile/injection.so",
        "LD_LIBRARY_PATH": "/hostile/libraries",
        "LD_AUDIT": "/hostile/audit.so",
        "CUDA_INJECTION64_PATH": "/hostile/nvbit.so",
        "CUDA_VISIBLE_DEVICES": "hostile",
        "PYTHONPATH": "/hostile/python",
        "BASH_ENV": "/hostile/bash-env",
        "NCU_CONFIG_FILE": "/hostile/ncu-config",
    })
    observed = {}

    def fake_run(
        command, *, cwd, env, timeout_s, log_dir, stem, failure_code,
        trusted_child_storage_marker=False, supervised_ncu_identity=None,
        supervised_ncu_environment=None,
    ):
        observed.update(
            command=command, cwd=cwd, env=env,
            supervised_ncu_identity=supervised_ncu_identity,
            supervised_ncu_environment=supervised_ncu_environment,
        )
        out = cell_dir / "labels" / "sm_86" / slug / shape
        out.mkdir(parents=True)
        label = json.loads(source_label.read_text())
        label.update({
            "device": "sm_86", "slug": slug, "shape_point": shape,
            "binary_sha256": invocation["executable_sha256"],
            "profiler_version": "Version 2026.1.0.0 (public-release)",
            "cuda_version": "13.2", "driver_version": "610.43.02",
            "measurement_protocol": {
                "ncu_clock_control": "none", "ncu_cache_control": "all",
                "ncu_replay_mode": "kernel", "host_persistence_mode": "Enabled",
                "host_sm_clock_mhz": 1350, "host_mem_clock_mhz": 10501,
            },
        })
        (out / "label_0_0.json").write_text(json.dumps(label) + "\n")
        (out / "label_run_manifest_0_0.json").write_text(json.dumps({
            "schema_version": 1, "source_commit": matrix["source_commit"],
            "binary_sha256": invocation["executable_sha256"],
            "kernel_signature_sha256": "1" * 64,
            "kernel_signature_source": "fallback_legacy", "arch": "sm_86",
            "slug": slug, "shape_point": shape,
            "repeat_index": 0, "launch_index": 0,
        }) + "\n")
        report = out / "run_0_0.ncu-rep"
        report.write_bytes(b"ncu report")
        _write_launch_observation(
            out / "run_0_0.launch_observation.json",
            report=report, launch_index=0,
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = log_dir / f"{stem}.stdout.log"
        stderr = log_dir / f"{stem}.stderr.log"
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        return 1.0, stdout, stderr

    monkeypatch.setattr(cc, "_run_logged", fake_run)
    ncu_identity = _expected_ncu_identity()
    result = cc.capture_label(
        cell=cell, binary=binary, invocation=invocation,
        cell_dir=cell_dir, matrix=matrix, pins=pins, timeout_s=10,
        expected_ncu_identity=ncu_identity,
        runtime_cwd=cc.REPO_ROOT, runtime_env=runtime_env,
        bind_supplemental_toolchain=True,
    )
    command = observed["command"]
    expected = [
        str(cc.LABEL_DRIVER), "--device", matrix["arch"],
        "--slug", slug,
        "--shape-point", shape,
        "--shape-binding", "logical-workload",
        "--repeat", "0", "--launch", "0",
        "--binary", str(binary),
        "--warmup", "1",
        "--ncu", ncu_identity.resolved_path,
        "--ncu-executable-sha256", ncu_identity.executable_sha256,
        "--ncu-version-line", ncu_identity.version_line,
        "--cache-control", "all",
        "--clock-control", "none",
        "--out-dir", str(cell_dir / "labels"),
        "--expected-cuda-release", "13.2",
        "--expected-ncu-version-prefix",
        pins["toolchain"]["ncu_version_prefix"],
        "--publication-metric-scope", "cycles_only_binding",
    ]
    for metric in matrix["metric_registry"]["collection"][
        "selected_unit_metrics"
    ]:
        expected += ["--unit-metric", metric]
    expected += [
        "--ncu-exec-path", cc.ncu_executor.AUTHORIZED_NCU_PROC_PATH,
        "--", *invocation["label_app_args"],
    ]
    assert command == expected
    separator = command.index("--")
    assert command[command.index("--shape-binding") + 1] == "logical-workload"
    assert command[command.index("--expected-cuda-release") + 1] == "13.2"
    assert command[command.index("--expected-ncu-version-prefix") + 1] == (
        pins["toolchain"]["ncu_version_prefix"]
    )
    assert command[command.index("--publication-metric-scope") + 1] == (
        "cycles_only_binding"
    )
    assert command[separator + 1:] == invocation["label_app_args"]
    assert not cc._shape_tokens(command[separator + 1:])
    assert observed["cwd"] == cc.REPO_ROOT
    assert observed["env"] is None
    assert observed["supervised_ncu_identity"] == ncu_identity
    collection_environment = observed["supervised_ncu_environment"]
    assert isinstance(
        collection_environment, cc.ncu_executor.NcuCollectionEnvironment,
    )
    projected = collection_environment.as_dict()
    assert projected["PATH"] == "/usr/local/cuda-13.2/bin:/usr/bin:/bin"
    assert projected["PWD"] == str(cc.REPO_ROOT)
    assert projected["CUTLASS_DIR"] == runtime_env["CUTLASS_DIR"]
    assert projected["LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE"] == "1"
    for name in (
        "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT",
        "CUDA_INJECTION64_PATH", "CUDA_VISIBLE_DEVICES", "PYTHONPATH",
        "BASH_ENV", "ENV", "HOME", "TMPDIR", "NCU_CONFIG_FILE",
    ):
        assert name not in projected
    capture_manifest = json.loads(result["capture_manifest"].read_text())
    assert capture_manifest["shape_binding"] == "logical-workload"
    assert capture_manifest["shape_forwarded_to_target"] is False
    assert capture_manifest["toolchain_contract"] == {
        "expected_cuda_release": {"major": 13, "minor": 2},
        "observed_cuda_release": {"major": 13, "minor": 2},
        "expected_ncu_version_prefix": pins["toolchain"][
            "ncu_version_prefix"
        ],
        "observed_ncu_version": "Version 2026.1.0.0 (public-release)",
    }
    assert capture_manifest["publication_metric_scope"] == {
        "decision": "CYCLES_ONLY_BINDING",
        "binding_metrics": ["gpc__cycles_elapsed.max"],
        "collection_scope": "registry_superset",
    }


def _write_sol_module_inventory(
        trace_dir: Path, *, module: dict, kernel_name: str = "candidate_kernel"):
    path = trace_dir / "extra_info" / "traced_kernel_modules.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "module_identity": {
            "stable_key": "sha256", "path_semantics": "path_at_capture",
            "rebind_contract": "sha256_required_basename_hint",
        },
        "selection": {
            "active_from_start": False,
            "max_accepted_active_launches": 1,
            "accepted_active_launches": 1,
            "active_launch_attempts": 1,
            "active_launches_skipped_after_limit": 0,
            "concurrent_limit_violations": 0,
            "exact_contract_satisfied": True,
            "module_mapping_count": 1,
            "inactive_channel_records_dropped": 0,
            "accepted_channel_records": 10,
        },
        "launches": [{
            "accepted_launch_index": 0, "device_id": 0, "stream_id": 0,
            "trace_launch_id": 1, "kernel_name": kernel_name,
            "resolution_source": "function_registry", "module": module,
        }],
    }) + "\n")


def _sol_control_trace_record(trace_dir: Path, log: Path) -> dict:
    return {
        "control_trace_dir": trace_dir, "stdout_log": log, "stderr_log": log,
        "launch_count": 1,
        "launches": [{
            "device_id": 0, "stream_id": 0, "kernel_id": 1,
            "kernel_name": "candidate_kernel___0",
        }],
    }


def test_runtime_sol_controls_require_exact_module_launch_and_dynamic_pcs(tmp_path):
    _, cell = _generated_sol_cell()
    cell_dir = tmp_path / "chunk" / "cells" / "sol__cell"
    trace_dir = cell_dir / "trace"
    module_path = cell_dir / "sol_stage" / "benchmark_kernel.so"
    module_path.parent.mkdir(parents=True)
    module_path.write_bytes(b"stable benchmark module")
    _write_sol_module_inventory(trace_dir, module={
        "status": "stable_static_module", "reason": "resolved_file",
        "path_at_capture": str(module_path.resolve()),
        "basename": module_path.name, "sha256": cc.sha256_file(module_path),
    })
    source = trace_dir / "extra_info" / "enhanced_execution_info.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps({
        "benchmark_name": "python",
        "kernels": [{
            "kernel_name": "candidate_kernel___0",
            "architecture_version": 86,
            "is_captured_from_binary": True,
            "unique_function_id": 7,
            "function_address": 4096,
            "instructions": [{
                "pc_num_dec": 0,
                "encoded_instruction": ["0x0000000000000000", "0x0000000000000000"],
                "control_bits": {
                    "stall_count": 1, "is_yield": False,
                    "is_new_read_barrier": False,
                    "is_new_write_barrier": True,
                    "id_new_read_barrier": 7,
                    "id_new_write_barrier": 0,
                    "wait_barrier_bits": 0,
                },
            }],
        }],
    }) + "\n")
    threadblock_pb2, _, _ = __import__(
        "import_trace_contract"
    )._import_pb_modules()
    tb = threadblock_pb2.threadblock()
    tb.warps[0].instructions.add().pc = 0
    tb_path = trace_dir / "threadblocks" / "device_0" / "0.pb"
    tb_path.parent.mkdir(parents=True)
    tb_path.write_bytes(tb.SerializeToString())
    log = cell_dir / "logs" / "trace.log"
    log.parent.mkdir(parents=True)
    log.write_text("")
    result = cc.capture_sol_control(
        cell=cell, invocation=cell["invocation"],
        label={}, trace=_sol_control_trace_record(trace_dir, log),
        cell_dir=cell_dir, module_store=tmp_path / "module_store",
    )
    assert result["outcome"] == bcm.SOL_CONTROL_OUTCOMES[0]
    assert result["total_decoded"] == 1
    payload = json.loads(result["control_bits"].read_text())
    assert payload["summary"]["kernel_count"] == 1
    kernel = payload["kernels"][0]
    assert kernel["control_source"] == "enhanced_runtime_binary"
    assert kernel["is_captured_from_binary"] is True
    assert kernel["instructions"] == [{
        "pc_num_dec": 0,
        "encoded_instruction": [
            "0x0000000000000000", "0x0000000000000000",
        ],
        "control_bits": {
            "stall_count": 1,
            "is_yield": False,
            "is_new_read_barrier": False,
            "is_new_write_barrier": True,
            "id_new_read_barrier": 7,
            "id_new_write_barrier": 0,
            "wait_barrier_bits": 0,
        },
    }]
    assert payload["static_rebind"] is None
    assert list((cell_dir / "control_evidence").glob("x-*.elf.bin")) == []


def _static_sol_control_case(tmp_path: Path) -> tuple[dict, dict, Path, Path, dict]:
    _, cell = _generated_sol_cell()
    cell_dir = tmp_path / "chunk" / "cells" / "sol__cell"
    trace_dir = cell_dir / "trace"
    module_path = cell_dir / "sol_stage" / "benchmark_kernel.so"
    module_path.parent.mkdir(parents=True)
    module_path.write_bytes(b"stable benchmark module")
    _write_sol_module_inventory(trace_dir, module={
        "status": "stable_static_module", "reason": "resolved_file",
        "path_at_capture": str(module_path.resolve()),
        "basename": module_path.name, "sha256": cc.sha256_file(module_path),
    })
    source = trace_dir / "extra_info" / "enhanced_execution_info.json"
    source.write_text(json.dumps({
        "benchmark_name": "python",
        "kernels": [{
            "kernel_name": "candidate_kernel___0",
            "architecture_version": 86,
            "is_captured_from_binary": False,
            "unique_function_id": 7,
            "function_address": 4096,
            "instructions": [
                {
                    "pc_num_dec": pc,
                    "pc_string_hex": f"{pc:x}",
                    "op_code": "MOV",
                    "encoded_instruction": [],
                    "control_bits": {
                        "stall_count": 0, "is_yield": False,
                        "is_new_read_barrier": False,
                        "is_new_write_barrier": False,
                        "id_new_read_barrier": 0,
                        "id_new_write_barrier": 0,
                        "wait_barrier_bits": 0,
                    },
                }
                for pc in (0, 16)
            ],
        }],
    }) + "\n")
    threadblock_pb2, _, _ = __import__(
        "import_trace_contract"
    )._import_pb_modules()
    tb = threadblock_pb2.threadblock()
    for pc in (0, 16):
        tb.warps[0].instructions.add().pc = pc
    tb_path = trace_dir / "threadblocks" / "device_0" / "0.pb"
    tb_path.parent.mkdir(parents=True)
    tb_path.write_bytes(tb.SerializeToString())
    log = cell_dir / "logs" / "trace.log"
    log.parent.mkdir(parents=True)
    log.write_text("")
    records = [
        {
            "pc": pc,
            "low_word": f"0x{pc + 1:016x}",
            "high_word": "0x0c0fe400078e00ff",
            "control": {
                "stall": 2, "yield_flag": 1,
                "write_barrier": 7, "read_barrier": 7,
                "wait_mask": 0, "reuse": 3,
            },
            "status": "decoded",
        }
        for pc in (0, 16)
    ]
    raw = b"".join(
        int(row["low_word"], 16).to_bytes(8, "little")
        + int(row["high_word"], 16).to_bytes(8, "little")
        for row in records
    )
    extracted = {
        "function_name": "candidate_kernel", "code_arch": "sm_86",
        "section_index": 7,
        "section_name": "x-candidate_kernel.sm_86.elf.bin",
        "raw_text_size_bytes": len(raw),
        "raw_text_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_text_bytes": raw, "records": records,
        "toolkit_version": "V13.2",
        "parser_contract": cc.sass_extractor.PARSER_CONTRACT,
    }
    report = cell_dir / "labels" / "run_0_0.ncu-rep"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"candidate_kernel\0" + raw)
    return cell, extracted, cell_dir, log, {"ncu_report": report}


def test_runtime_sol_controls_static_rebind_ignores_placeholder_controls(
    tmp_path, monkeypatch,
):
    cell, extracted, cell_dir, log, label = _static_sol_control_case(tmp_path)
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *args, **kwargs: copy.deepcopy(extracted),
    )
    result = cc.capture_sol_control(
        cell=cell, invocation=cell["invocation"], label=label,
        trace=_sol_control_trace_record(cell_dir / "trace", log),
        cell_dir=cell_dir, module_store=tmp_path / "module_store",
    )
    assert result["outcome"] == bcm.SOL_CONTROL_OUTCOMES[1]
    assert result["total_decoded"] == 2
    payload = json.loads(result["control_bits"].read_text())
    kernel = payload["kernels"][0]
    assert kernel["is_captured_from_binary"] is False
    assert kernel["control_source"] == (
        "raw_static_text_section_runtime_rebind"
    )
    assert kernel["instructions"][0]["encoded_instruction"] == [
        "0x0000000000000001", "0x0c0fe400078e00ff",
    ]
    assert kernel["instructions"][0]["control_bits"] == {
        "id_new_read_barrier": 7,
        "id_new_write_barrier": 7,
        "is_new_read_barrier": False,
        "is_new_write_barrier": False,
        "is_yield": True,
        "reuse_flags": 3,
        "stall_count": 2,
        "wait_barrier_bits": 0,
    }
    rebind = payload["static_rebind"]
    assert rebind["raw_text_sha256"] == extracted["raw_text_sha256"]
    assert rebind["static_instruction_count"] == 2
    assert rebind["ncu_report_raw_text_occurrences"] == 1
    assert (cell_dir / rebind["raw_text_path"]).read_bytes() == extracted["raw_text_bytes"]


def test_runtime_sol_controls_accepts_unique_cubin_symbol_raw_match(
    tmp_path, monkeypatch,
):
    cell, extracted, cell_dir, log, label = _static_sol_control_case(tmp_path)
    function = extracted["function_name"]
    alias = "x-candidate-1.sm_86.elf.bin"
    extracted["section_name"] = alias
    extracted["section_identity_method"] = "cubin_symbol_raw_match"
    extracted["cubin_symbol_proof"] = {
        "symbol_index": 3,
        "symbol_table_index": 3,
        "text_section_index": 17,
        "text_section_name": f".text.{function}",
        "code_arch": "sm_86",
        "cubin_catalog_index": 1,
        "cubin_file_name": "candidate.1.sm_86.cubin",
        "cubin_sha256": "a" * 64,
        "catalog_raw_match_count": 1,
    }
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *args, **kwargs: copy.deepcopy(extracted),
    )
    result = cc.capture_sol_control(
        cell=cell, invocation=cell["invocation"], label=label,
        trace=_sol_control_trace_record(cell_dir / "trace", log),
        cell_dir=cell_dir, module_store=tmp_path / "module_store",
    )
    assert result["outcome"] == bcm.SOL_CONTROL_OUTCOMES[1]
    payload = json.loads(result["control_bits"].read_text())
    rebind = payload["static_rebind"]
    assert rebind["text_section_identity_method"] == "cubin_symbol_raw_match"
    assert rebind["text_section_name"] == alias
    assert rebind["cubin_symbol_proof"] == extracted["cubin_symbol_proof"]
    assert (cell_dir / rebind["raw_text_path"]).read_bytes() == extracted["raw_text_bytes"]


def test_runtime_sol_controls_rejects_nonunique_cubin_symbol_raw_match_proof(
    tmp_path, monkeypatch,
):
    cell, extracted, cell_dir, log, label = _static_sol_control_case(tmp_path)
    function = extracted["function_name"]
    extracted["section_name"] = "x-candidate-1.sm_86.elf.bin"
    extracted["section_identity_method"] = "cubin_symbol_raw_match"
    extracted["cubin_symbol_proof"] = {
        "symbol_index": 3,
        "symbol_table_index": 3,
        "text_section_index": 17,
        "text_section_name": f".text.{function}",
        "code_arch": "sm_86",
        "cubin_catalog_index": 1,
        "cubin_file_name": "candidate.1.sm_86.cubin",
        "cubin_sha256": "a" * 64,
        "catalog_raw_match_count": 2,
    }
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *args, **kwargs: copy.deepcopy(extracted),
    )
    with pytest.raises(
        cc.ChunkRefusal, match="sol_control_static_contract_mismatch",
    ):
        cc.capture_sol_control(
            cell=cell, invocation=cell["invocation"], label=label,
            trace=_sol_control_trace_record(cell_dir / "trace", log),
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )


@pytest.mark.parametrize("raw_occurrences", [0, 2])
def test_runtime_sol_static_rebind_requires_exact_same_cell_ncu_section(
    tmp_path, monkeypatch, raw_occurrences,
):
    cell, extracted, cell_dir, log, label = _static_sol_control_case(tmp_path)
    label["ncu_report"].write_bytes(
        b"candidate_kernel\0" + extracted["raw_text_bytes"] * raw_occurrences
    )
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *args, **kwargs: copy.deepcopy(extracted),
    )
    with pytest.raises(cc.ChunkRefusal, match="sol_control_ncu_report_binding_mismatch"):
        cc.capture_sol_control(
            cell=cell, invocation=cell["invocation"], label=label,
            trace=_sol_control_trace_record(cell_dir / "trace", log),
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )


def test_runtime_sol_static_rebind_requires_exact_descriptor_pcs(
    tmp_path, monkeypatch,
):
    cell, extracted, cell_dir, log, label = _static_sol_control_case(tmp_path)
    extracted["records"][1]["pc"] = 32
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *args, **kwargs: copy.deepcopy(extracted),
    )
    with pytest.raises(
        cc.ChunkRefusal, match="sol_control_static_descriptor_binding_invalid"
    ):
        cc.capture_sol_control(
            cell=cell, invocation=cell["invocation"], label=label,
            trace=_sol_control_trace_record(cell_dir / "trace", log),
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )


@pytest.mark.parametrize("mutation", ["function", "raw_digest", "decoded_control"])
def test_runtime_sol_static_rebind_rejects_inconsistent_extractor_result(
    tmp_path, monkeypatch, mutation,
):
    cell, extracted, cell_dir, log, label = _static_sol_control_case(tmp_path)
    if mutation == "function":
        extracted["function_name"] = "wrong_function"
    elif mutation == "raw_digest":
        extracted["raw_text_sha256"] = "0" * 64
    else:
        extracted["records"][0]["control"]["stall"] ^= 1
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *args, **kwargs: copy.deepcopy(extracted),
    )
    with pytest.raises(cc.ChunkRefusal, match="sol_control_static_contract_mismatch"):
        cc.capture_sol_control(
            cell=cell, invocation=cell["invocation"], label=label,
            trace=_sol_control_trace_record(cell_dir / "trace", log),
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )


def test_runtime_sol_static_rebind_rejects_empty_descriptor_opcode(
    tmp_path, monkeypatch,
):
    cell, extracted, cell_dir, log, label = _static_sol_control_case(tmp_path)
    source = cell_dir / "trace" / "extra_info" / "enhanced_execution_info.json"
    document = json.loads(source.read_text())
    document["kernels"][0]["instructions"][0]["op_code"] = ""
    source.write_text(json.dumps(document) + "\n")
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *args, **kwargs: copy.deepcopy(extracted),
    )
    with pytest.raises(
        cc.ChunkRefusal, match="sol_control_static_descriptor_binding_invalid"
    ):
        cc.capture_sol_control(
            cell=cell, invocation=cell["invocation"], label=label,
            trace=_sol_control_trace_record(cell_dir / "trace", log),
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )


def test_runtime_sol_static_rebind_rejects_nonboolean_capture_source(tmp_path):
    cell, _, cell_dir, log, label = _static_sol_control_case(tmp_path)
    source = cell_dir / "trace" / "extra_info" / "enhanced_execution_info.json"
    document = json.loads(source.read_text())
    del document["kernels"][0]["is_captured_from_binary"]
    source.write_text(json.dumps(document) + "\n")
    with pytest.raises(cc.ChunkRefusal, match="sol_control_kernel_identity_invalid"):
        cc.capture_sol_control(
            cell=cell, invocation=cell["invocation"], label=label,
            trace=_sol_control_trace_record(cell_dir / "trace", log),
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )


def test_runtime_sol_static_rebind_rejects_extraction_ambiguity(
    tmp_path, monkeypatch,
):
    cell, _, cell_dir, log, label = _static_sol_control_case(tmp_path)

    def refuse(*args, **kwargs):
        raise cc.sass_extractor.FunctionTextExtractionError("two exact sections")

    monkeypatch.setattr(cc.sass_extractor, "extract_exact_function_text", refuse)
    with pytest.raises(cc.ChunkRefusal, match="sol_control_static_extraction_failed"):
        cc.capture_sol_control(
            cell=cell, invocation=cell["invocation"], label=label,
            trace=_sol_control_trace_record(cell_dir / "trace", log),
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )


def test_runtime_sol_static_rebind_requires_dynamic_pc_subset(
    tmp_path, monkeypatch,
):
    cell, extracted, cell_dir, log, label = _static_sol_control_case(tmp_path)
    threadblock_pb2, _, _ = __import__(
        "import_trace_contract"
    )._import_pb_modules()
    tb = threadblock_pb2.threadblock()
    tb.warps[0].instructions.add().pc = 32
    (cell_dir / "trace" / "threadblocks" / "device_0" / "0.pb").write_bytes(
        tb.SerializeToString()
    )
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *args, **kwargs: copy.deepcopy(extracted),
    )
    with pytest.raises(cc.ChunkRefusal, match="sol_control_dynamic_pc_unbound"):
        cc.capture_sol_control(
            cell=cell, invocation=cell["invocation"], label=label,
            trace=_sol_control_trace_record(cell_dir / "trace", log),
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )
    assert not (cell_dir / "control_bits.json").exists()
    assert not (cell_dir / "control_capture_manifest.json").exists()
    assert list((cell_dir / "control_evidence").glob("x-*.elf.bin")) == []


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("path_at_capture", "relative/module.so", "sol_control_module_identity_invalid"),
        ("basename", "wrong-module.so", "sol_control_module_identity_invalid"),
        ("sha256", "0" * 64, "input_hash_mismatch"),
    ],
)
def test_preserve_sol_module_rejects_path_basename_and_hash_mismatch(
    tmp_path, field, value, error,
):
    module_path = tmp_path / "module.so"
    module_path.write_bytes(b"stable module")
    module = {
        "path_at_capture": str(module_path.resolve()),
        "basename": module_path.name,
        "sha256": cc.sha256_file(module_path),
    }
    module[field] = value
    with pytest.raises(cc.ChunkRefusal, match=error):
        cc._preserve_sol_module(
            module,
            cell_dir=tmp_path / "cell",
            module_store=(tmp_path / "module_store").resolve(),
        )


def test_runtime_sol_controls_only_type_producer_declared_unstable_module(tmp_path):
    _, cell = _generated_sol_cell()
    cell_dir = tmp_path / "chunk" / "cells" / "sol__cell"
    trace_dir = cell_dir / "trace"
    _write_sol_module_inventory(trace_dir, module={
        "status": "no_stable_static_module",
        "reason": "jit_or_anonymous_mapping", "path_at_capture": "",
        "basename": "", "sha256": "",
    })
    log = cell_dir / "logs" / "trace.log"
    log.parent.mkdir(parents=True)
    log.write_text("")
    result = cc.capture_sol_control(
        cell=cell, invocation=cell["invocation"],
        label={}, trace=_sol_control_trace_record(trace_dir, log),
        cell_dir=cell_dir, module_store=tmp_path / "module_store",
    )
    assert result["outcome"] == "typed_unavailable_no_stable_static_module"
    assert result["typed_reason"] == "jit_or_anonymous_mapping"


def _vendor_control_fixture(
    tmp_path: Path, *, fallback: bool,
) -> tuple[dict, dict, dict, dict, Path, bytes]:
    cell_dir = tmp_path / "chunk" / "cells" / "simbench_bw__vendor"
    trace_dir = cell_dir / "trace"
    function = "ampere_sgemm_128x32_nn" if fallback else "ampere_bf16_test"
    runtime_kernel = f"{function}___0"
    module_path = tmp_path / "runtime" / "libcublasLt.so.test"
    module_path.parent.mkdir(parents=True)
    module_path.write_bytes(b"pinned runtime module")
    module_sha = cc.sha256_file(module_path)
    nvcc = tmp_path / "toolkit" / "nvcc"
    cuobjdump = tmp_path / "toolkit" / "cuobjdump"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_bytes(b"nvcc")
    cuobjdump.write_bytes(b"cuobjdump")
    raw = b"R" * 32
    records = [
        {
            "pc": pc, "low_word": f"0x{pc + 1:016x}",
            "high_word": "0x000fe400078e00ff",
            "control": {
                "stall": 2, "yield_flag": 1, "write_barrier": 7,
                "read_barrier": 7, "wait_mask": 0, "reuse": 0,
            },
            "status": "decoded",
        }
        for pc in (0, 16)
    ]
    contract = {
        "policy": "post_trace_vendor_module",
        "module_discovery": "nvbit_runtime_module_inventory",
        "selection_contract": "active_from_start_single_launch_exact",
        "runtime_kernel": runtime_kernel,
        "function_name": function,
        "execution_arch": "sm_86", "code_arch": "sm_80",
        "inventory_binding": (
            "ncu_report_exact_text_section" if fallback
            else "nvbit_stable_static_module"
        ),
        "expected_outcome": (
            "matrix_pinned_runtime_module_controls_emitted" if fallback
            else "runtime_controls_emitted"
        ),
        "expected_inventory_status": (
            "no_stable_static_module" if fallback else "stable_static_module"
        ),
        "expected_inventory_reason": "origin_unresolved" if fallback else "resolved_file",
        "module_path_at_capture": str(module_path.resolve()),
        "module_basename": module_path.name, "module_sha256": module_sha,
        "text_section_index": 7,
        "text_section_name": f"x-{function}.sm_80.elf.bin",
        "raw_text_size_bytes": len(raw),
        "raw_text_sha256": hashlib.sha256(raw).hexdigest(),
        "static_instruction_count": 2,
        "static_pcs_sha256": cc.canonical_sha256([0, 16]),
        "ncu_report_binding": "raw_text_section_exactly_once",
        "toolkit_version": "V13.2",
        "nvcc_path": str(nvcc), "nvcc_sha256": cc.sha256_file(nvcc),
        "cuobjdump_path": str(cuobjdump),
        "cuobjdump_sha256": cc.sha256_file(cuobjdump),
    }
    cell = {
        "identity": {
            "roster_source": "simbench_bw",
            "slug": "vendor_sgemm_cublas" if fallback else "vendor_gemm_cublas",
        },
        "control_bits": contract,
    }
    invocation = {
        "authority_reference_path": "vendor_run",
        "authority_reference_sha256": "a" * 64,
    }
    module = (
        {
            "status": "no_stable_static_module", "reason": "origin_unresolved",
            "path_at_capture": "", "basename": "", "sha256": "",
        }
        if fallback else {
            "status": "stable_static_module", "reason": "resolved_file",
            "path_at_capture": str(module_path.resolve()),
            "basename": module_path.name, "sha256": module_sha,
        }
    )
    inventory_path = trace_dir / "extra_info" / "traced_kernel_modules.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps({
        "schema_version": 1,
        "module_identity": {
            "stable_key": "sha256", "path_semantics": "path_at_capture",
            "rebind_contract": "sha256_required_basename_hint",
        },
        "selection": {
            "active_from_start": True, "max_accepted_active_launches": 0,
            "accepted_active_launches": 1, "active_launch_attempts": 1,
            "active_launches_skipped_after_limit": 0,
            "concurrent_limit_violations": 0, "exact_contract_satisfied": True,
            "module_mapping_count": 1, "inactive_channel_records_dropped": 0,
            "accepted_channel_records": 10,
        },
        "launches": [{
            "accepted_launch_index": 0, "device_id": 0, "stream_id": 0,
            "trace_launch_id": 1, "kernel_name": function,
            "resolution_source": "origin_unresolved" if fallback else "function_registry",
            "module": module,
        }],
    }) + "\n")
    enhanced = trace_dir / "extra_info" / "enhanced_execution_info.json"
    enhanced.write_text(json.dumps({
        "kernels": [{
            "kernel_name": runtime_kernel, "architecture_version": 86,
            "is_captured_from_binary": False, "unique_function_id": 7,
            "function_address": 4096,
            "instructions": [
                {
                    "pc_num_dec": pc, "pc_string_hex": f"{pc:x}",
                    "op_code": "MOV", "encoded_instruction": [],
                    "control_bits": {},
                }
                for pc in (0, 16)
            ],
        }],
    }) + "\n")
    threadblock_pb2, _, _ = __import__("import_trace_contract")._import_pb_modules()
    tb = threadblock_pb2.threadblock()
    for pc in (0, 16):
        tb.warps[0].instructions.add().pc = pc
    tb_path = trace_dir / "threadblocks" / "device_0" / "0.pb"
    tb_path.parent.mkdir(parents=True)
    tb_path.write_bytes(tb.SerializeToString())
    log = cell_dir / "logs" / "trace.log"
    log.parent.mkdir(parents=True)
    log.write_text("")
    report = cell_dir / "labels" / "run_0_0.ncu-rep"
    report.parent.mkdir(parents=True)
    report.write_bytes(function.encode() + b"\0" + raw)
    label = {"ncu_report": report}
    trace = {
        "control_trace_dir": trace_dir, "stdout_log": log, "stderr_log": log,
        "launch_count": 1,
        "launches": [{
            "device_id": 0, "stream_id": 0, "kernel_id": 1,
            "kernel_name": runtime_kernel,
        }],
    }
    extracted = {
        "function_name": function, "code_arch": "sm_80",
        "section_index": 7, "section_name": contract["text_section_name"],
        "raw_text_size_bytes": len(raw),
        "raw_text_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_text_bytes": raw, "records": records,
        "toolkit_version": "V13.2", "parser_contract": {"control_layout": {}},
    }
    return cell, invocation, label, trace, cell_dir, extracted


@pytest.mark.parametrize("fallback", [False, True])
def test_vendor_runtime_controls_bind_exact_static_section_and_dynamic_pcs(
    tmp_path, monkeypatch, fallback,
):
    cell, invocation, label, trace, cell_dir, extracted = _vendor_control_fixture(
        tmp_path, fallback=fallback,
    )
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *_args, **_kwargs: copy.deepcopy(extracted),
    )
    result = cc.capture_vendor_control(
        cell=cell, invocation=invocation, label=label, trace=trace,
        cell_dir=cell_dir, module_store=tmp_path / "module_store",
    )
    assert result["outcome"] == cell["control_bits"]["expected_outcome"]
    assert result["total_decoded"] == 2
    payload = json.loads(result["control_bits"].read_text())
    assert payload["execution_arch"] == "sm_86"
    assert payload["code_arch"] == "sm_80"
    assert payload["ncu_report_raw_text_occurrences"] == 1
    assert payload["kernels"][0]["dynamic_trace_pc_count"] == 2
    assert payload["kernels"][0]["instructions"][0]["control_bits"][
        "stall_count"
    ] == 2


def test_vendor_runtime_controls_refuse_missing_ncu_section_binding(tmp_path, monkeypatch):
    cell, invocation, label, trace, cell_dir, extracted = _vendor_control_fixture(
        tmp_path, fallback=True,
    )
    label["ncu_report"].write_bytes(cell["control_bits"]["function_name"].encode())
    monkeypatch.setattr(
        cc.sass_extractor, "extract_exact_function_text",
        lambda *_args, **_kwargs: copy.deepcopy(extracted),
    )
    with pytest.raises(cc.ChunkRefusal, match="ncu_report_binding_mismatch"):
        cc.capture_vendor_control(
            cell=cell, invocation=invocation, label=label, trace=trace,
            cell_dir=cell_dir, module_store=tmp_path / "module_store",
        )


def _certified_sol_probe(matrix: dict, cell: dict) -> dict:
    pin = copy.deepcopy(
        matrix["input_pins"]["sol_native_env_fingerprint"]
    )
    certified = json.loads((cc.REPO_ROOT / pin["path"]).read_text())
    environment = copy.deepcopy(
        cell["invocation"]["adapter_contract"]["environment"]
    )
    return {
        "status": cc.preflight.STATUS_OK,
        "typed_error": None,
        "detail": "synthetic no-GPU certified environment probe",
        "ambient": {},
        "environment_contract": environment,
        "environment_contract_sha256": cc._sol_certifying_environment(
            cell["invocation"]
        )[1],
        "constructed_environment_delta": dict(CERTIFYING_SOL_ENVIRONMENT),
        "matrix_pin": pin,
        "matrix_sha256": MATRIX_SHA,
        "observed_fingerprint": {
            "components": certified["components"],
            "hash": pin["native_env_hash"],
            "missing": [],
            "degraded": False,
        },
        "compiler_identity": {
            "requested": "g++-14",
            "resolved_realpath": "/usr/bin/g++-14",
            "executable_sha256": "4" * 64,
            "version_line": "g++-14 synthetic no-GPU compiler",
        },
        "nvcc_release": {"major": 13, "minor": 2},
        "nvcc_version_output": "Cuda compilation tools, release 13.2",
    }


def test_capture_cell_sol_typed_route_seals_full_orchestration_without_gpu(
        tmp_path, monkeypatch):
    matrix, cell = _generated_sol_cell()
    invocation = cell["invocation"]
    cell_dir = (
        tmp_path / "chunk" / "cells"
        / f"sol_execbench__{cell['identity']['slug']}"
    )
    cell_dir.mkdir(parents=True)
    probe = _certified_sol_probe(matrix, cell)
    live_pin = copy.deepcopy(probe["matrix_pin"])
    matrix["input_pins"]["sol_native_env_fingerprint"] = {
        "path": "data/local_labels/campaign/sol_native_env_fingerprint_sm86.json",
        "sha256": "0" * 64,
        "native_env_hash": "1" * 64,
    }
    certified_env = _certified_sol_env(cell)
    budget = SimpleNamespace(
        max_trace_bytes=100,
        max_wall_time_s=10,
        sampling_policy="off",
    )
    provenance = {
        "protocol_version": "v2.1",
        "source_commit": "d" * 40,
        "capture_matrix_sha256": "5" * 64,
        "preflight_sha256": "6" * 64,
        "capture_started_at_utc": "2026-07-10T12:00:00Z",
        "admission_run_id": "9" * 64,
        "supersedes": None,
        "prior_preparation_failures": [],
        "prior_capture_defect_attempts": [],
        "pins_sha256": "7" * 64,
        "trace_budget_sha256": "8" * 64,
        "raw_staging_margin_bytes": 1 << 20,
    }
    events: list[str] = []
    state: dict = {}

    monkeypatch.setattr(cc, "require_cell_budget", lambda *_: budget)

    def fake_guard(_pins):
        event = f"guard_{sum(item.startswith('guard_') for item in events) + 1}"
        events.append(event)
        return {"status": "ok", "event": event}

    monkeypatch.setattr(cc, "live_capture_guard", fake_guard)

    def fake_prepare_run(
            command, *, cwd, env, timeout_s, log_dir, stem, failure_code):
        assert (cell_dir / "sol_native_env_fingerprint.json").is_file()
        request_path = Path(env["SOL_EXECBENCH_CAPTURE_REQUEST"])
        request = json.loads(request_path.read_text())
        assert request["action"] == "prepare"
        assert request["environment_contract"] == CERTIFYING_SOL_ENVIRONMENT
        assert request["environment_contract_sha256"] == (
            cc._sol_certifying_environment(invocation)[1]
        )
        _write_valid_sol_stage(Path(request["stage_dir"]), cell, invocation)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = log_dir / f"{stem}.stdout.log"
        stderr = log_dir / f"{stem}.stderr.log"
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        events.append("prepare")
        return 0.01, stdout, stderr

    monkeypatch.setattr(cc, "_run_logged", fake_prepare_run)
    real_prepare = cc.prepare_sol_capture

    def tracked_prepare(**kwargs):
        context = real_prepare(**kwargs)
        state["context"] = context
        return context

    monkeypatch.setattr(cc, "prepare_sol_capture", tracked_prepare)

    def write_log(stem: str) -> tuple[Path, Path]:
        root = cell_dir / "logs"
        root.mkdir(parents=True, exist_ok=True)
        stdout = root / f"{stem}.stdout.log"
        stderr = root / f"{stem}.stderr.log"
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        return stdout, stderr

    def fake_label(**kwargs):
        assert kwargs["execution_source_commit"] == provenance["source_commit"]
        assert kwargs["kernel_name_selector"] is None
        context = state["context"]
        runtime_env = kwargs["runtime_env"]
        request = json.loads(
            Path(runtime_env["SOL_EXECBENCH_CAPTURE_REQUEST"]).read_text()
        )
        assert request["action"] == "execute"
        assert request["capture_kind"] == "label"
        _write_valid_sol_phase_evidence(context, runtime_env)
        phase = kwargs["phase_result_validator"](runtime_env, "label")
        label_root = cell_dir / "labels" / "synthetic"
        label_root.mkdir(parents=True)
        sidecar = label_root / "label_0_0.json"
        run_manifest = label_root / "label_run_manifest_0_0.json"
        ncu_report = label_root / "run_0_0.ncu-rep"
        ncu_launch_observation = (
            label_root / "run_0_0.launch_observation.json"
        )
        sidecar.write_text('{"synthetic":"label"}\n')
        run_manifest.write_text('{"synthetic":"run"}\n')
        ncu_report.write_bytes(b"synthetic ncu report")
        _write_launch_observation(
            ncu_launch_observation,
            report=ncu_report, launch_index=0,
            kernel_name="candidate_kernel",
        )
        stdout, stderr = write_log("label")
        capture_manifest = cell_dir / "logs" / "label_capture_manifest.json"
        capture_manifest.write_text(json.dumps({
            "schema_version": 1,
            "sol_phase_evidence": phase,
        }) + "\n")
        events.append("label")
        return {
            "command": [str(cc.LABEL_DRIVER), "--"],
            "sidecar": sidecar,
            "run_manifest": run_manifest,
            "ncu_report": ncu_report,
            "ncu_launch_observation": ncu_launch_observation,
            "ncu_launch_observation_document": json.loads(
                ncu_launch_observation.read_text()
            ),
            "stdout_log": stdout,
            "stderr_log": stderr,
            "capture_manifest": capture_manifest,
            "sol_phase_evidence": phase,
            "unit_metric_count": len(
                matrix["metric_registry"]["collection"][
                    "selected_unit_metrics"
                ]
            ),
        }

    monkeypatch.setattr(cc, "capture_label", fake_label)

    def fake_trace(**kwargs):
        context = state["context"]
        full_env = kwargs["runtime_env"]
        full_request = json.loads(
            Path(full_env["SOL_EXECBENCH_CAPTURE_REQUEST"]).read_text()
        )
        assert full_request["capture_kind"] == "trace"
        assert full_request["environment_contract"] == CERTIFYING_SOL_ENVIRONMENT
        _write_valid_sol_phase_evidence(context, full_env)
        full_phase = kwargs["phase_result_validator"](full_env, "trace")

        counting_env = kwargs["phase_env_factory"](
            "trace_sampling_counting_v2", 1.0
        )
        counting_request = json.loads(
            Path(counting_env["SOL_EXECBENCH_CAPTURE_REQUEST"]).read_text()
        )
        assert counting_request["capture_kind"] == "trace"
        _write_valid_sol_phase_evidence(context, counting_env)
        counting_phase = kwargs["phase_result_validator"](
            counting_env, "trace"
        )

        raw = cell_dir / "trace_full_raw"
        (raw / "threadblocks" / "device_0").mkdir(parents=True)
        (raw / "dynamic_trace.pb").write_bytes(b"synthetic root")
        (raw / "stats.csv").write_text("synthetic stats\n")
        (raw / "threadblocks" / "device_0" / "0.pb").write_bytes(
            b"synthetic raw threadblock"
        )
        _write_sol_module_inventory(raw, module={
            "status": "no_stable_static_module",
            "reason": "jit_or_anonymous_mapping",
            "path_at_capture": "",
            "basename": "",
            "sha256": "",
        })

        retained = cell_dir / "trace"
        retained.mkdir()
        (retained / "trace_sampling.json").write_text(json.dumps({
            "schema_version": 2,
            "method": "unavailable",
            "synthetic": True,
        }) + "\n")
        (retained / "ac9_unavailable_evidence.json").write_text(json.dumps({
            "schema_version": 1,
            "kind": "ac9_typed_unavailable_evidence",
            "typed_reason": "gate_forced",
            "preflight_sha256": provenance["preflight_sha256"],
        }) + "\n")

        trace_manifest = cell_dir / "trace_capture_manifest.json"
        trace_manifest.write_text(json.dumps({
            "schema_version": 1,
            "identity": cell["identity"],
            "sol_phase_evidence": full_phase,
            "counting_phase_evidence": counting_phase,
            "launch_count": 1,
        }) + "\n")
        route_result = {
            "schema_version": 1,
            "kind": "ac9_trace_budget_route",
            "cell_id": cell["identity"]["slug"],
            "identity": cell["identity"],
            "protocol": {
                **provenance,
                "capture_manifest_sha256": cc.sha256_file(trace_manifest),
            },
            "outcome": "typed_unavailable",
            "typed_reason": "gate_forced",
            "method": "unavailable",
            "retained": {
                "tree_sha256": cc.canonical_sha256("synthetic retained"),
                "bytes": cc._trace_tree_bytes(retained),
            },
        }
        route_result["result_sha256"] = cc.budget_route.canonical_sha256(
            route_result
        )
        route_path = cell_dir / "trace_budget_route.json"
        cc.budget_route.write_route_result(route_result, route_path)
        stdout, stderr = write_log("trace")
        events.append("trace_routed")
        return {
            "manifest": trace_manifest,
            "route_result_path": route_path,
            "route_result": route_result,
            "trace_dir": retained,
            "control_trace_dir": raw,
            "raw_trace_dir": raw,
            "root_pb": None,
            "stats_csv": None,
            "stdout_log": stdout,
            "stderr_log": stderr,
            "routed": True,
            "outcome": "typed_unavailable",
            "typed_reason": "gate_forced",
            "threadblocks_pb_count": 1,
            "admitted_records": 1,
            "launch_count": 1,
            "launches": [{
                "device_id": 0,
                "stream_id": 0,
                "kernel_id": 1,
                "kernel_name": "candidate_kernel___0",
                "block_dim": [128, 1, 1],
                "grid_dim": [1, 1, 1],
            }],
        }

    monkeypatch.setattr(cc, "capture_trace", fake_trace)

    real_control = cc.capture_sol_control

    def tracked_control(**kwargs):
        raw = kwargs["trace"]["control_trace_dir"]
        assert raw == cell_dir / "trace_full_raw"
        assert raw.is_dir()
        assert not (cell_dir / "control_capture_manifest.json").exists()
        events.append("control_started_with_raw")
        result = real_control(**kwargs)
        assert raw.is_dir()
        assert result["control_bits"].is_file()
        events.append("control_completed_with_raw")
        return result

    monkeypatch.setattr(cc, "capture_sol_control", tracked_control)
    real_cleanup = cc._remove_routed_raw_trace

    def tracked_cleanup(trace, *, cell_dir):
        raw = trace["raw_trace_dir"]
        assert raw.is_dir()
        assert (cell_dir / "control_bits.json").is_file()
        assert (cell_dir / "control_capture_manifest.json").is_file()
        events.append("cleanup_after_control")
        real_cleanup(trace, cell_dir=cell_dir)
        assert not raw.exists()
        events.append("raw_deleted")

    monkeypatch.setattr(cc, "_remove_routed_raw_trace", tracked_cleanup)

    result = cc.capture_cell(
        cell=cell,
        cell_dir=cell_dir,
        matrix=matrix,
        pins=json.loads(cc.DEFAULT_PINS.read_text()),
        cell_budgets=object(),
        expected_tracer_sha256="9" * 64,
        capture_provenance=provenance,
        expected_ncu_identity=_expected_ncu_identity(),
        certified_sol_env=certified_env,
        sol_environment_probe=probe,
        module_store=(tmp_path / "module_store").resolve(),
        expected_sol_fingerprint_pin=live_pin,
    )

    assert result["disposition"] == "failed"
    assert result["typed_reason"] == "trace_unsupported"
    assert "method='unavailable'" in result["detail"]
    assert "reason='gate_forced'" in result["detail"]
    assert events == [
        "guard_1",
        "prepare",
        "guard_2",
        "label",
        "guard_3",
        "trace_routed",
        "control_started_with_raw",
        "control_completed_with_raw",
        "cleanup_after_control",
        "raw_deleted",
        "guard_4",
    ]
    assert events.index("control_started_with_raw") < events.index(
        "control_completed_with_raw"
    ) < events.index("cleanup_after_control") < events.index("raw_deleted")
    assert not (cell_dir / "trace_full_raw").exists()
    attestation = json.loads(
        (cell_dir / "sol_native_env_fingerprint.json").read_text()
    )
    assert attestation["matrix_pin"] == live_pin
    assert attestation["certified_native_env_hash"] == live_pin[
        "native_env_hash"
    ]

    attestation_path = cell_dir / "sol_native_env_fingerprint.json"
    attestation = json.loads(attestation_path.read_text())
    attestation_seal = attestation.pop("attestation_sha256")
    assert attestation_seal == cc.sol_env.canonical_sha256(attestation)
    assert attestation["certified_hash_equal"] is True
    assert attestation["observed_native_env_hash"] == live_pin[
        "native_env_hash"
    ]

    request_files = sorted(
        path.name for path in (cell_dir / "sol_requests").glob("*.json")
    )
    assert request_files == [
        "label.json",
        "prepare.json",
        "trace_full.json",
        "trace_sampling_counting_v2.json",
    ]
    for request_path in (cell_dir / "sol_requests").glob("*.json"):
        request = json.loads(request_path.read_text())
        assert request["environment_contract"] == CERTIFYING_SOL_ENVIRONMENT
        assert request["environment_contract_sha256"] == (
            cc._sol_certifying_environment(invocation)[1]
        )

    route_path = cell_dir / "trace_budget_route.json"
    sealed_route = json.loads(route_path.read_text())
    route_seal = sealed_route.pop("result_sha256")
    assert route_seal == cc.budget_route.canonical_sha256(sealed_route)
    assert sealed_route["protocol"]["preflight_sha256"] == (
        provenance["preflight_sha256"]
    )
    cell_result = json.loads((cell_dir / "cell_capture_result.json").read_text())
    assert cell_result["trace_route_result_sha256"] == route_seal
    assert cell_result["trace_route_outcome"] == "typed_unavailable"
    assert cell_result["control_outcome"] == (
        "typed_unavailable_no_stable_static_module"
    )

    kinds = {artifact["artifact_kind"] for artifact in result["artifacts"]}
    assert kinds == {
        "cell_result",
        "control_bits",
        "control_evidence",
        "control_manifest",
        "execution_log",
            "label_run_manifest",
            "label_sidecar",
            "ncu_launch_observation",
            "ncu_report",
        "sol_capture_evidence",
        "trace_auxiliary",
        "trace_manifest",
        "trace_route_record",
    }
    relative_paths = [
        artifact["relative_path"] for artifact in result["artifacts"]
    ]
    assert len(relative_paths) == len(set(relative_paths))
    assert any(path.endswith("sol_native_env_fingerprint.json")
               for path in relative_paths)
    assert any("/sol_stage/evidence/label-result.json" in path
               for path in relative_paths)
    assert any("/sol_requests/trace_full.json" in path
               for path in relative_paths)
    assert all("trace_full_raw" not in path for path in relative_paths)
    route_artifact = next(
        artifact for artifact in result["artifacts"]
        if artifact["artifact_kind"] == "trace_route_record"
    )
    assert route_artifact["sha256"] == cc.sha256_file(route_path)
