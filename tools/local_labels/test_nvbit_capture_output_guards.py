"""Capture/output guard coverage: exactly two tests per inventory row."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tools.local_labels import activation_trace_orchestrator as activation
from tools.local_labels import campaign_preflight as campaign
from tools.local_labels import roster_path_resolver as resolver
from tools.local_labels import roster_trace_capture as roster_capture
from tools.local_labels import sm120_activation_bundle_producer as sm120
from tools.local_labels import stage_measurement_manifests_from_predictor_trace as staging
from tools.local_labels import trace_register_cap_variant_cells as variants
from tools.local_labels.canary import emit_canary_manifest as canary
from tools.local_labels.protected_paths import ProtectedPathError, is_protected_path


REPO = Path(__file__).resolve().parents[2]
STAGING = REPO / "results" / "nvbit_capture_staging"
PROTECTED = REPO / "data" / "local_labels" / "campaign"
GOLDEN_SCRIPT = REPO / "tools" / "trace_sampling" / "golden_cells" / "capture_golden_cells.sh"


def _arm_zero_mutation(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Instrument every mutation/launch family a refusal must precede."""

    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"mutation occurred before clearance: {name}")
        return fail

    monkeypatch.setattr(Path, "mkdir", forbidden("mkdir"))
    monkeypatch.setattr(Path, "write_text", forbidden("write_text"))
    monkeypatch.setattr(Path, "unlink", forbidden("unlink"))
    monkeypatch.setattr(shutil, "rmtree", forbidden("rmtree"))
    monkeypatch.setattr(os, "replace", forbidden("replace"))
    monkeypatch.setattr(tempfile, "mkdtemp", forbidden("mkdtemp"))
    monkeypatch.setattr(subprocess, "run", forbidden("subprocess"))
    return calls


def test_activation_orchestrator_safe_output_plan() -> None:
    plan = activation._activation_output_plan(
        slug="layernorm_fwd_bf16_sm80",
        shape_point="timing_shape",
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        fallback_source="portable_lcd",
        output_root=STAGING / "test_activation",
    )
    assert all(path.is_relative_to(STAGING) for path in plan.values())
    assert plan["portable_manifest"].name == "manifest.json"


def test_activation_orchestrator_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        activation._activation_output_plan(
            slug="layernorm_fwd_bf16_sm80",
            shape_point="timing_shape",
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            fallback_source="portable_lcd",
            output_root=PROTECTED / "activation",
        )
    assert calls == []


def test_roster_capture_safe_batch_plan() -> None:
    plans, manifest = roster_capture._roster_capture_plan(
        roster=[{
            "slug": "layernorm_fwd_bf16_sm80",
            "shape_point": "timing_shape",
            "app_args": ["--timing_shape"],
        }],
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=STAGING / "test_roster",
        manifest_out=None,
    )
    assert manifest.is_relative_to(STAGING)
    assert plans[0]["portable"] is True
    assert Path(plans[0]["traces_folder"]).is_relative_to(STAGING)


def test_roster_capture_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        roster_capture._roster_capture_plan(
            roster=[], sku="rtx_3090_ti", arch_dir="sm_86",
            output_root=PROTECTED / "roster", manifest_out=None,
        )
    assert calls == []


def test_register_cap_safe_output_plan() -> None:
    plan = variants._variant_output_plan(
        slug="register_spill_local_mem",
        shape_point="trace_l2_accuracy_xl",
        register_cap=64,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=STAGING / "test_variants",
    )
    assert all(path.is_relative_to(STAGING) for path in plan.values())
    assert plan["tmp_path"].name.endswith(".tmp")


def test_register_cap_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        variants._variant_output_plan(
            slug="register_spill_local_mem",
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            output_root=PROTECTED / "variants",
        )
    assert calls == []


def test_golden_capture_safe_plan_is_nonmutating(tmp_path: Path) -> None:
    output = STAGING / "test_golden_plan"
    existed = output.exists()
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "NVBIT_CAPTURE_OUTPUT_ROOT": str(output),
    }
    proc = subprocess.run(
        [str(GOLDEN_SCRIPT), "--plan-only", "gs_stream_copy"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert output.exists() is existed


def test_golden_capture_protected_plan_has_zero_mutation() -> None:
    forbidden = PROTECTED / "guard_negative_must_not_exist"
    assert not forbidden.exists()
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "NVBIT_CAPTURE_OUTPUT_ROOT": str(forbidden),
    }
    proc = subprocess.run(
        [str(GOLDEN_SCRIPT), "--plan-only", "gs_stream_copy"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert not forbidden.exists()


def test_canary_safe_output_plan() -> None:
    plan = canary._canary_output_plan(
        out=STAGING / "test_canary" / "manifest.json",
        work=STAGING / "test_canary" / "work",
    )
    assert len([key for key in plan if key.endswith("_binary")]) == 3
    assert len([key for key in plan if key.endswith("_trace")]) == 3
    assert all(path.is_relative_to(STAGING) for path in plan.values())


def test_canary_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        canary._canary_output_plan(
            out=STAGING / "test_canary" / "manifest.json",
            work=PROTECTED / "canary",
        )
    assert calls == []


def test_campaign_preflight_safe_output_plan(tmp_path: Path) -> None:
    output, smoke = campaign._campaign_output_plan(
        output=STAGING / "test_campaign" / "preflight.json",
        smoke_temp_root=tmp_path / "smoke",
    )
    assert output.is_relative_to(STAGING)
    assert all(not is_protected_path(path, repo_root=REPO) for path in smoke.values())
    assert not smoke["root"].exists()


def test_campaign_preflight_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        campaign._campaign_output_plan(output=PROTECTED / "preflight.json")
    assert calls == []


def test_measurement_stager_safe_mirrored_output_plan() -> None:
    source = (
        REPO / "workloads" / "portable_lcd" / "results" / "slug"
        / "rtx_3090_ti" / "sm_86"
    )
    target = staging._staging_artifact_dir(
        source, output_root=STAGING / "test_manifests",
    )
    assert target.is_relative_to(STAGING)
    assert target.parts[-6:] == source.parts[-6:]


def test_measurement_stager_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        staging._staging_artifact_dir(
            REPO / "workloads" / "portable_lcd" / "results" / "slug",
            output_root=PROTECTED / "manifests",
        )
    assert calls == []


def test_sm120_bundle_safe_output_plan(tmp_path: Path) -> None:
    plan = sm120._bundle_output_plan(
        candidates=[("register_spill_local_mem", "trace_l2_accuracy_xl")],
        output_root=STAGING / "test_sm120",
        helper_output=tmp_path / "sm120_helper",
    )
    cell = plan["cells"][("register_spill_local_mem", "trace_l2_accuracy_xl")]
    assert plan["output_root"].is_relative_to(STAGING)
    assert all(path.is_relative_to(STAGING) for path in cell.values())
    assert not plan["helper"].is_relative_to(REPO)


def test_sm120_bundle_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        sm120._bundle_output_plan(
            candidates=[], output_root=PROTECTED / "sm120",
        )
    assert calls == []


def test_roster_resolver_returns_read_only_authority_path() -> None:
    path = resolver.resolve_cell_root(
        slug="layernorm_fwd_bf16_sm80",
        shape_point="timing_shape",
        sku="rtx_3090_ti",
        arch_dir="sm_86",
    )
    assert is_protected_path(path, repo_root=REPO)
    assert "write" not in resolver.__dict__


def test_roster_resolver_invalid_fallback_has_zero_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ValueError):
        resolver.resolve_cell_root(
            slug="not_in_roster",
            shape_point="not_a_shape",
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            fallback_source="not_a_source",
        )
    assert calls == []
