from __future__ import annotations

import copy
import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).with_name("m5_source_surface_registry.json")
SCHEMA_PATH = (
    REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "m5_source_surface_registry.v1.schema.json"
)

CLOSURE_ROOTS = (
    "CLAUDE.md",
    "arch_profiles",
    "data/local_labels",
    "data/pchase",
    "docs",
    "eval/frozen_labels",
    "eval/guards",
    "plans",
    "sim",
    "tools/local_labels",
    "tools/nvbit_tracer",
    "tools/pchase",
    "tools/producer_retirement.py",
    "tools/sass_control",
    "workloads/simbench_bw",
)

EXPLICIT_SURFACES = {
    "CLAUDE.md",
    "arch_profiles/__init__.py",
    "arch_profiles/cards/sm_86.json",
    "arch_profiles/loader.py",
    "arch_profiles/schema/arch_profile.schema.json",
    "arch_profiles/tests/test_arch_profile_lockstep.py",
    "data/local_labels/campaign/capture_roster_authority_sm86.json",
    "data/local_labels/campaign/sm86_reference_capture_matrix.json",
    "data/local_labels/nvml_reference_operating_point_sm_86.json",
    "data/local_labels/nvml_short_burst_anchor_sm_86.json",
    "docs/STATUS.md",
    "docs/immutable_publication_contract.md",
    "eval/guards/test_targets_not_features_guard.py",
    "eval/frozen_labels/sm86_m1_bundle_v21.json",
    "plans/fable5_consultation_and_response/02-immutable-publication--final.merged.prompt.txt",
    "plans/fable5_consultation_and_response/02-immutable-publication.fable5.response.txt",
    "plans/micro2027_ncu_sim_execution_plan.md",
    "plans/sm86_calibration_microbench_design_fable5_answer.md",
    "plans/sm86_calibration_microbench_design_fable5_prompt.md",
    "plans/sm86_comprehensive_live_capture_campaign_plan.md",
    "plans/sm86_cp_async_timing_probe_fable5_answer.md",
    "plans/sm86_cp_async_timing_probe_fable5_prompt.md",
    "sim/calib/src/lib.rs",
    "sim/mem/src/l1.rs",
    "sim/mem/src/lib.rs",
    "sim/mem/src/pchase_card.rs",
    "sim/sm/src/issue.rs",
    "sim/sm/tests/machine_local_replay.rs",
    "tools/local_labels/calibrate_nvml_reference_operating_point.py",
    "tools/local_labels/calibrate_nvml_short_burst_anchor.py",
    "tools/local_labels/campaign_protocol_pins.json",
    "tools/local_labels/fit_calibration_head.py",
    "tools/local_labels/fit_calibration_head_v4_sm120.py",
    "tools/local_labels/fit_l1_calibration_head.py",
    "tools/local_labels/fit_monotonic_calibration_head_v5.py",
    "tools/local_labels/immutable_artifact.py",
    "tools/local_labels/sm120_calibration_head_historical_only_scanner.py",
    "tools/local_labels/test_calibration_head_v5_schema.py",
    "tools/local_labels/test_immutable_artifact.py",
    "tools/local_labels/test_source_s_retirement_guards.py",
    "tools/nvbit_tracer/nvbit_release/core/nvbit.h",
    "tools/producer_retirement.py",
    "tools/sass_control/emit_control_bits_sidecar.py",
    "tools/sass_control/extractor.py",
    "tools/sass_control/run_pc_join_witness.py",
    "workloads/simbench_bw/nvml_covariate_acceptance.toml",
    "workloads/simbench_bw/nvml_covariate_join.py",
    "workloads/simbench_bw/nvml_metadata_verifier.py",
    "workloads/simbench_bw/schemas/calibration_evaluation_v1.schema.json",
    "workloads/simbench_bw/schemas/calibration_head_v1.schema.json",
    "workloads/simbench_bw/schemas/calibration_head_v5.schema.json",
    "workloads/simbench_bw/schemas/nvml_covariate.schema.json",
    "workloads/simbench_bw/schemas/nvml_reference_operating_point.schema.json",
    "workloads/simbench_bw/schemas/sass_control_pc_join_witness.schema.json",
}

AUTO_IDENTITY_PATHS = {
    "data/local_labels/campaign/capture_roster_authority_sm86.json",
    "data/local_labels/campaign/sm86_reference_capture_matrix.json",
    "data/pchase/sm_86_static_card.json",
    "tools/local_labels/campaign_protocol_pins.json",
    "tools/local_labels/immutable_artifact.py",
    "tools/local_labels/m5_calibration_controller.py",
    "tools/local_labels/m5_trust_anchors.json",
    "tools/pchase/pchase_kernels.cu",
    "tools/sass_control/emit_control_bits_sidecar.py",
    "tools/sass_control/extractor.py",
    "tools/sass_control/run_pc_join_witness.py",
    "workloads/simbench_bw/schemas/m5_calibration_done.v2.schema.json",
    "workloads/simbench_bw/schemas/m5_calibration_result.v2.schema.json",
    "workloads/simbench_bw/schemas/m5_calibration_run.v2.schema.json",
    "workloads/simbench_bw/schemas/m5_quantity_authority.v2.schema.json",
    "workloads/simbench_bw/schemas/m5_trust_anchor_registry.v1.schema.json",
    "workloads/simbench_bw/schemas/sass_control_pc_join_witness.schema.json",
}

M5_TOKEN = re.compile(rb"(?<![A-Za-z0-9])(?:M5|m5)(?=$|[^A-Za-z0-9])")
CODE_SCAN_ROOTS = (
    "arch_profiles",
    "docs",
    "eval/guards",
    "plans",
    "sim",
    "tools/local_labels",
    "workloads/simbench_bw/schemas",
)


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=reject_duplicates)
    assert isinstance(value, dict)
    return value, payload


def _available_paths() -> set[str]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *CLOSURE_ROOTS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return {
        raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    }


def _discovered_surfaces(available: set[str]) -> set[str]:
    discovered = set(EXPLICIT_SURFACES)
    for path in available:
        if re.search(r"(?:^|[/_.-])m5(?:[_.-]|$)", path.lower()):
            discovered.add(path)
        if path.startswith(("data/pchase/", "tools/pchase/")):
            discovered.add(path)
        if path.startswith("workloads/simbench_bw/schemas/") and any(
            token in Path(path).name
            for token in ("calibration_evaluation", "calibration_head", "pchase")
        ):
            discovered.add(path)
        if path.startswith(CODE_SCAN_ROOTS):
            candidate = REPO_ROOT / path
            if candidate.is_file() and M5_TOKEN.search(candidate.read_bytes()):
                discovered.add(path)
    return discovered


def _closure_difference(
    registered: set[str], discovered: set[str]
) -> tuple[set[str], set[str]]:
    return discovered - registered, registered - discovered


def test_registry_is_closed_schema_valid_and_non_authoritative() -> None:
    registry, payload = _strict_json(REGISTRY_PATH)
    schema, _schema_payload = _strict_json(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(registry)

    assert payload == (
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert registry["authority_policy"] == (
        "NO_SOURCE_SURFACE_IS_AN_M5_AUTHORITY"
    )
    assert registry["collection_gate"] == "M5_COLLECT_DISABLED"
    assert registry["card_write_policy"] == "FORBIDDEN"
    assert registry["credit_state"] == (
        "NO_REVIEW_OPERATOR_HARDWARE_OR_CAMPAIGN_CREDIT"
    )
    assert registry["external_gate_state"] == (
        "REVIEW_OPERATOR_HARDWARE_UNCHANGED"
    )
    assert registry["milestone_state"] == "M5_NOT_STARTED"
    assert tuple(registry["closure_roots"]) == CLOSURE_ROOTS

    entries = registry["entries"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    assert all(entry["authority_candidate"] is False for entry in entries)
    assert all(entry["can_collect"] is False for entry in entries)

    available = _available_paths()
    registered = set(paths)
    assert registered <= available
    for path in sorted(registered):
        candidate = REPO_ROOT / path
        assert not candidate.is_symlink(), path
        assert candidate.is_file(), path
    unknown, stale = _closure_difference(
        registered, _discovered_surfaces(available)
    )
    assert unknown == set(), f"M5_INVENTORY_CLOSURE_ABSENT unknown={sorted(unknown)}"
    assert stale == set(), f"M5_INVENTORY_CLOSURE_ABSENT stale={sorted(stale)}"
    assert AUTO_IDENTITY_PATHS <= registered


def test_unclassified_m5_surface_fails_closure() -> None:
    registry, _payload = _strict_json(REGISTRY_PATH)
    registered = {entry["path"] for entry in registry["entries"]}
    sentinel = (
        REPO_ROOT
        / "tools"
        / "local_labels"
        / f"m5_unclassified_{uuid.uuid4().hex}.py"
    )
    relative = sentinel.relative_to(REPO_ROOT).as_posix()
    sentinel.write_text("# synthetic closure sentinel\n", encoding="utf-8")
    try:
        available = _available_paths()
        discovered = _discovered_surfaces(available)
        assert relative in available
        assert relative in discovered
        unknown, stale = _closure_difference(registered, discovered)
        assert relative in unknown
        assert stale == set()
    finally:
        sentinel.unlink()


def test_quarantined_ancestry_cannot_be_reclassified_active() -> None:
    registry, _payload = _strict_json(REGISTRY_PATH)
    schema, _schema_payload = _strict_json(SCHEMA_PATH)
    quarantined = {
        "plans/fable5_consultation_and_response/01-m5-rule1-provenance--final.merged.prompt.txt",
        "plans/fable5_consultation_and_response/01-m5-rule1-provenance.fable5.response.txt",
        "plans/sm86_calibration_microbench_design_fable5_answer.md",
        "plans/sm86_calibration_microbench_design_fable5_prompt.md",
        "plans/sm86_cp_async_timing_probe_fable5_answer.md",
        "plans/sm86_cp_async_timing_probe_fable5_prompt.md",
        "workloads/simbench_bw/schemas/m5_calibration_done.v1.schema.json",
        "workloads/simbench_bw/schemas/m5_calibration_result.v1.schema.json",
        "workloads/simbench_bw/schemas/m5_calibration_run.v1.schema.json",
    }
    by_path = {entry["path"]: entry for entry in registry["entries"]}
    assert all(by_path[path]["lifecycle"] == "quarantined" for path in quarantined)

    for path in sorted(quarantined):
        mutated = copy.deepcopy(registry)
        next(
            entry for entry in mutated["entries"] if entry["path"] == path
        )["lifecycle"] = "active"
        errors = list(
            jsonschema.Draft202012Validator(schema).iter_errors(mutated)
        )
        assert any(error.validator == "const" for error in errors), path


def test_active_campaign_plan_cannot_cleanse_quarantined_ancestry() -> None:
    registry, _payload = _strict_json(REGISTRY_PATH)
    schema, _schema_payload = _strict_json(SCHEMA_PATH)
    mutated = copy.deepcopy(registry)
    plan = next(
        entry
        for entry in mutated["entries"]
        if entry["path"]
        == "plans/sm86_comprehensive_live_capture_campaign_plan.md"
    )
    plan["reason"] = "Clean M5 preregistration authority."

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(mutated))

    assert errors, "M5_RULE1_PREREGISTRATION_ANCESTRY"
    assert any(error.validator == "const" for error in errors)


def test_calibration_head_cannot_be_reclassified_as_m5_producer() -> None:
    registry, _payload = _strict_json(REGISTRY_PATH)
    schema, _schema_payload = _strict_json(SCHEMA_PATH)
    mutated = copy.deepcopy(registry)
    substitute = next(
        entry
        for entry in mutated["entries"]
        if entry["path"] == "tools/local_labels/fit_calibration_head.py"
    )
    substitute["category"] = "producer"

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(mutated))

    assert errors, "M5_SUBSTITUTE_IDENTITY_FORBIDDEN"
    assert any(error.validator == "const" for error in errors)


def test_no_registered_source_can_self_promote_to_authority() -> None:
    registry, _payload = _strict_json(REGISTRY_PATH)
    schema, _schema_payload = _strict_json(SCHEMA_PATH)
    mutated = copy.deepcopy(registry)
    controller = next(
        entry
        for entry in mutated["entries"]
        if entry["path"]
        == "tools/local_labels/m5_calibration_controller.py"
    )
    controller["authority_candidate"] = True

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(mutated))

    assert errors, "M5_SOURCE_SURFACE_AUTHORITY_FORBIDDEN"
    assert any(error.validator == "const" for error in errors)
