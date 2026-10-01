"""Task-23 source-S retirement, provenance, and frozen-byte guards."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import build_sm86_capture_matrix as matrix_builder  # noqa: E402
from tools.local_labels import capture_long_timing_cohort as long_cohort  # noqa: E402
from tools.local_labels import capture_timing_labels as timing_labels  # noqa: E402
from tools.local_labels import emit_timing_prediction_contract as timing_contract  # noqa: E402
from tools.pchase import run_pchase_card  # noqa: E402
from tools.pchase import transcribe_latencies_to_arch_card as transcribe  # noqa: E402
from tools.producer_retirement import (  # noqa: E402
    LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE,
    RETIRED_PRODUCER_EXIT_CODE,
    LegacyProducerRetiredError,
)

_OLD_REGISTRY_SHA256 = (
    "be1f7893cf653800c69f6e39a4d22abfae4d67c10197e7d5f9bd46c6900c2423"
)
_ACTIVE_REGISTRY_SHA256 = (
    "95d90f8c4fffd9376b8d785047a9e02ecafc81ed43da6081985958c2b9dec62d"
)
_FROZEN_MATRIX_SHA256 = (
    "af29d72e955cf42e281036190f2d55d72af0af92a632c78c8d159073a072fb0e"
)

_FROZEN_ARTIFACT_SHA256 = {
    "data/local_labels/campaign/sm86_reference_capture_matrix.json": (
        _FROZEN_MATRIX_SHA256
    ),
    "data/pchase/sm_86_static_card.json": (
        "e87d2e208f5326a8083d1a54799864953a62ed1fece1b628583cf8d5cafc581d"
    ),
    "data/local_labels/timing_prediction_contract_sm_86.json": (
        "b00af638bfb70cbdd48fd6226744136ea3549ccd2ed407f720dce5435435bbdf"
    ),
    "data/annex/sm86_sweeps_2026H1/timing_prediction_contract_sm_86.json": (
        "b00af638bfb70cbdd48fd6226744136ea3549ccd2ed407f720dce5435435bbdf"
    ),
    "workloads/simbench_bw/schemas/timing_prediction_contract.schema.json": (
        "326071c7cda75eec69d38c38ec3bb48e3d0022a2d93b1bcfdf0111f2b962ff78"
    ),
    "data/local_labels/timing_cohort/ac4_long_cohort_manifest.json": (
        "10e30d61516e158747d066d67d4011077b9091966fedfa107d8f37c6e111df04"
    ),
    "data/local_labels/timing_cohort/ac4_long_duration_cohort_sm_86.json": (
        "abe3d9f600e19a0d3a3f812f271cb4ac4fe3fbd144d56adae3646f2e19214d49"
    ),
    "data/local_labels/timing_cohort/ac4_long_cohort_exclusions_sm_86.json": (
        "4d3f3d0bf6a5e6e40896c74d4fdd926657b77346edbe9fd52352abb94c818f6a"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbid_side_effect(*_args, **_kwargs):
    raise AssertionError("retired producer reached file or subprocess side effect")


def _block_file_and_subprocess_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    for method in (
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "mkdir",
        "is_file",
        "exists",
        "replace",
        "rename",
        "unlink",
    ):
        monkeypatch.setattr(Path, method, _forbid_side_effect)
    for method in ("run", "check_output", "Popen"):
        monkeypatch.setattr(subprocess, method, _forbid_side_effect)


def _assert_typed_retirement(exc: LegacyProducerRetiredError) -> None:
    assert exc.reason == LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE
    assert exc.reason in str(exc)
    assert exc.surface
    assert exc.successor_requirement


def test_capture_timing_callable_and_cli_retire_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    label_path = Path("must-not-be-read.json")
    _block_file_and_subprocess_side_effects(monkeypatch)

    with pytest.raises(LegacyProducerRetiredError) as caught:
        timing_labels.capture_and_land(
            label_admitted_path=label_path,
            ncu="must-not-be-checked",
            repeats=5,
        )
    _assert_typed_retirement(caught.value)

    with pytest.raises(LegacyProducerRetiredError) as caught:
        timing_labels._ncu_capture_once(
            "must-not-be-checked", Path("must-not-be-read"), []
        )
    _assert_typed_retirement(caught.value)

    assert timing_labels.main(["--help"]) == RETIRED_PRODUCER_EXIT_CODE
    assert LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE in capsys.readouterr().err


def test_long_cohort_callable_and_cli_retire_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _block_file_and_subprocess_side_effects(monkeypatch)

    with pytest.raises(LegacyProducerRetiredError) as caught:
        long_cohort.capture_cohort(
            manifest={},
            ncu="must-not-be-checked",
            repeats=5,
            floor_ns=551760,
            source_commit="0" * 40,
        )
    _assert_typed_retirement(caught.value)

    with pytest.raises(LegacyProducerRetiredError) as caught:
        long_cohort._ncu_capture_once(
            "must-not-be-checked", Path("must-not-be-read"), []
        )
    _assert_typed_retirement(caught.value)

    assert long_cohort.main(["--help"]) == RETIRED_PRODUCER_EXIT_CODE
    assert LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE in capsys.readouterr().err


def test_timing_contract_emitter_and_cli_retire_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = Path("must-not-be-written.json")
    monkeypatch.setattr(
        timing_contract._contract,
        "validate_contract",
        _forbid_side_effect,
    )
    _block_file_and_subprocess_side_effects(monkeypatch)

    with pytest.raises(LegacyProducerRetiredError) as caught:
        timing_contract.emit_contract(
            out_path=out_path,
            source_commit="0" * 40,
        )
    _assert_typed_retirement(caught.value)

    assert timing_contract.main(["--help"]) == RETIRED_PRODUCER_EXIT_CODE
    assert LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE in capsys.readouterr().err


def test_pchase_v1_main_retires_before_build_nvml_probe_or_write(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("build_binary", "nvml_sample", "run_probe"):
        monkeypatch.setattr(run_pchase_card, name, _forbid_side_effect)
    _block_file_and_subprocess_side_effects(monkeypatch)

    assert run_pchase_card.main(["--quick"]) == RETIRED_PRODUCER_EXIT_CODE
    diagnostic = capsys.readouterr().err
    assert LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE in diagnostic
    assert "schema-v2 external immutable evidence" in diagnostic


def test_v1_transcriber_retires_before_card_reads_or_writes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _block_file_and_subprocess_side_effects(monkeypatch)

    assert transcribe.main(["--host-clock-note", "arbitrary"]) == (
        RETIRED_PRODUCER_EXIT_CODE
    )
    diagnostic = capsys.readouterr().err
    assert LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE in diagnostic
    assert "verified schema-v2 protocol metadata" in diagnostic


@pytest.mark.parametrize(
    ("relative_script", "successor_requirement"),
    (
        (
            "tools/pchase/run_pchase_card.py",
            "schema-v2 external immutable evidence",
        ),
        (
            "tools/pchase/transcribe_latencies_to_arch_card.py",
            "verified schema-v2 protocol metadata",
        ),
    ),
)
def test_retired_pchase_direct_cli_reports_typed_refusal(
    relative_script: str,
    successor_requirement: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, relative_script, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == RETIRED_PRODUCER_EXIT_CODE
    assert completed.stdout == ""
    assert LEGACY_PRODUCER_RETIRED_IMMUTABLE_EVIDENCE in completed.stderr
    assert successor_requirement in completed.stderr
    assert "Traceback" not in completed.stderr


def test_historical_v1_contract_builder_remains_exact_and_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_path = (
        REPO_ROOT / "data/local_labels/timing_prediction_contract_sm_86.json"
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    real_datetime = dt.datetime

    class FrozenDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            value = real_datetime(2026, 5, 30, 18, 47, 41, tzinfo=dt.timezone.utc)
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(timing_contract._dt, "datetime", FrozenDateTime)
    rebuilt = timing_contract._build_historical_v1_contract(
        source_commit=frozen["source_commit"]
    )

    assert rebuilt == frozen
    assert not hasattr(timing_contract, "build_contract")
    assert rebuilt["schema_version"] == 1


def test_frozen_historical_artifact_bytes_remain_exact() -> None:
    for relative, expected in _FROZEN_ARTIFACT_SHA256.items():
        assert _sha256(REPO_ROOT / relative) == expected, relative


def test_registry_annotation_transition_preserves_matrix_v1_pin() -> None:
    registry_path = REPO_ROOT / matrix_builder.METRIC_REGISTRY_PATH
    matrix_path = (
        REPO_ROOT
        / "data/local_labels/campaign/sm86_reference_capture_matrix.json"
    )
    active_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    frozen_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    assert active_registry["registry_version"] == 2
    assert active_registry["label_schema_version"] == 5
    assert _sha256(registry_path) == _ACTIVE_REGISTRY_SHA256
    assert _sha256(matrix_path) == _FROZEN_MATRIX_SHA256
    assert frozen_matrix["input_pins"]["metric_registry"]["sha256"] == (
        _OLD_REGISTRY_SHA256
    )

    assert matrix_builder.validate_matrix(
        frozen_matrix,
        REPO_ROOT,
        validation_mode="frozen_v1",
    ) == []
    assert _ACTIVE_REGISTRY_SHA256 != _OLD_REGISTRY_SHA256
    print(f"active_metric_registry_sha256={_ACTIVE_REGISTRY_SHA256}")


_POSITIVE_INVARIANCE = re.compile(
    r"\b(?:DVFS[- ]stable|boost[- ]invariant)\b",
    re.IGNORECASE,
)
_NEGATOR = re.compile(r"\b(?:not|never|neither|without)\b", re.IGNORECASE)
_ACTIVE_CLAIM_PATHS = (
    "arch_profiles/schema/arch_profile.schema.json",
    "tools/arch_capabilities.json",
    "tools/local_labels/capture_timing_labels.py",
    "tools/local_labels/capture_long_timing_cohort.py",
    "tools/local_labels/emit_timing_prediction_contract.py",
    "tools/local_labels/metric_registry.json",
    "tools/pchase/run_pchase_card.py",
    "tools/pchase/transcribe_latencies_to_arch_card.py",
    "arch_profiles/cards/sm_86.json",
    "docs/PREDICTIVE_CLOSURE.md",
    "sim/arch/src/card.rs",
)
_HISTORICAL_FUNCTION_EXCLUSIONS = {
    "tools/local_labels/emit_timing_prediction_contract.py": {
        "_build_historical_v1_contract"
    },
}


def _historical_function_lines(relative: str, text: str) -> set[int]:
    names = _HISTORICAL_FUNCTION_EXCLUSIONS.get(relative, set())
    if not names:
        return set()
    tree = ast.parse(text, filename=relative)
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    assert set(found) == names, f"missing historical exclusion in {relative}"
    lines: set[int] = set()
    for node in found.values():
        assert node.end_lineno is not None
        lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def _positive_claims(relative: str, text: str) -> list[str]:
    excluded = _historical_function_lines(relative, text)
    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line_number in excluded:
            continue
        for match in _POSITIVE_INVARIANCE.finditer(line):
            prefix = line[max(0, match.start() - 96):match.start()]
            if _NEGATOR.search(prefix):
                continue
            findings.append(f"{relative}:{line_number}:{line.strip()}")
    return findings


def test_active_source_authorities_make_no_positive_invariance_claims() -> None:
    findings: list[str] = []
    for relative in _ACTIVE_CLAIM_PATHS:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        findings.extend(_positive_claims(relative, text))
    assert findings == []

    assert _positive_claims("sample", "cycles are boost-invariant")
    assert not _positive_claims("sample", "cycles are not boost-invariant")
