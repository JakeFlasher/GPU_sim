"""Contracts for the independent NCU launch-identity observation."""

from __future__ import annotations

import subprocess

import pytest

from tools.local_labels import parse_ncu_metrics as parser


def test_report_import_requests_mangled_kernel_names(monkeypatch, tmp_path):
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(b"synthetic report bytes")
    observed: dict[str, list[str]] = {}
    csv_text = (
        '"ID","Kernel Name","Block Size","Grid Size","Device"\n'
        '"0","_Z15syntheticKernelv","(128, 1, 1)",'
        '"(1, 1, 1)","0"\n'
    )

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "check": False,
        }
        return subprocess.CompletedProcess(command, 0, csv_text, "")

    monkeypatch.setattr(parser.subprocess, "run", fake_run)
    imported, error = parser._import_report_csv("/opt/ncu", report)

    assert error is None
    assert imported == csv_text
    command = observed["command"]
    assert command.count("--print-kernel-base") == 1
    option = command.index("--print-kernel-base")
    assert command[option + 1] == "mangled"
    observation = parser.extract_launch_observation(imported)
    assert observation == {
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": "0",
            "Device": "0",
            "Kernel Name": "_Z15syntheticKernelv",
            "Block Size": "(128, 1, 1)",
            "Grid Size": "(1, 1, 1)",
        },
    }


_REQUIRED_IDENTITY = {
    "ID": "0",
    "Kernel Name": "_Z15syntheticKernelv",
    "Block Size": "(128, 1, 1)",
    "Grid Size": "(1, 1, 1)",
    "Device": "0",
}


def _observation_csv(*, omit: str | None = None, blank: str | None = None) -> str:
    fields = [name for name in _REQUIRED_IDENTITY if name != omit]
    values = {
        name: "" if name == blank else _REQUIRED_IDENTITY[name]
        for name in fields
    }
    return (
        ",".join(f'"{name}"' for name in fields)
        + "\n"
        + ",".join(f'"{values[name]}"' for name in fields)
        + "\n"
    )


@pytest.mark.parametrize("field", tuple(_REQUIRED_IDENTITY))
def test_launch_observation_rejects_missing_mandatory_column(field):
    with pytest.raises(ValueError, match="identity columns missing"):
        parser.extract_launch_observation(_observation_csv(omit=field))


@pytest.mark.parametrize("field", tuple(_REQUIRED_IDENTITY))
def test_launch_observation_rejects_duplicate_mandatory_column(field):
    fields = [*_REQUIRED_IDENTITY, field]
    values = [*(_REQUIRED_IDENTITY.values()), "forged"]
    text = (
        ",".join(f'"{name}"' for name in fields)
        + "\n"
        + ",".join(f'"{value}"' for value in values)
        + "\n"
    )
    with pytest.raises(ValueError, match="identity columns duplicate"):
        parser.extract_launch_observation(text)


@pytest.mark.parametrize("field", tuple(_REQUIRED_IDENTITY))
def test_launch_observation_rejects_blank_mandatory_value(field):
    with pytest.raises(ValueError, match="identity values blank"):
        parser.extract_launch_observation(_observation_csv(blank=field))


def test_launch_observation_rejects_multiple_ncu_identities():
    text = _observation_csv() + (
        '"1","_Z15syntheticKernelv","(128, 1, 1)",'
        '"(1, 1, 1)","0"\n'
    )
    with pytest.raises(ValueError, match="identity cardinality is 2"):
        parser.extract_launch_observation(text)
