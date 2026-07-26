"""Regression tests for the count-label capture CLI boundary."""

from __future__ import annotations

from pathlib import Path

from tools.local_labels import capture_count_labels as capture


def test_dry_run_never_enters_ncu_capture(monkeypatch, capsys) -> None:
    calls: list[dict] = []

    def fail_if_called(**kwargs):
        calls.append(kwargs)
        raise AssertionError("dry-run reached count capture")

    monkeypatch.setattr(capture, "capture_and_land", fail_if_called)
    output_paths = [
        capture.REPO_ROOT / "data/local_labels/test-dry-run-labels.json",
        capture.REPO_ROOT / "data/local_labels/test-dry-run-exclusions.json",
        capture.REPO_ROOT / "data/local_labels/test-dry-run-governance.json",
        capture.REPO_ROOT / "data/local_labels/test-dry-run-audit.json",
    ]
    result = capture.main(
        [
            "--label-admitted",
            str(output_paths[0]),
            "--exclusions-out",
            str(output_paths[1]),
            "--governance-out",
            str(output_paths[2]),
            "--audit-out",
            str(output_paths[3]),
            "--ncu",
            "/does/not/exist/ncu",
            "--dry-run",
        ]
    )

    assert result == 0
    assert calls == []
    assert capsys.readouterr().err == "[capture_counts] --dry-run: no files written\n"
    assert all(not path.exists() for path in output_paths)
