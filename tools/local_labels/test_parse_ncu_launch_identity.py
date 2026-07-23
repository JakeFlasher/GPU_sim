"""Contracts for the independent NCU launch-identity observation."""

from __future__ import annotations

import subprocess

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
    assert observation["identity"]["Kernel Name"] == "_Z15syntheticKernelv"
