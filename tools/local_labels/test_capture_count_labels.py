"""Regression tests for the count-label capture CLI boundary."""

from __future__ import annotations

import pytest

from tools.local_labels import capture_count_labels as capture
from tools.local_labels import capture_t2_stream_labels as t2


def test_t2_dry_run_never_enters_ncu_capture(monkeypatch, capsys) -> None:
    calls: list[dict] = []

    def fail_if_called(**kwargs):
        calls.append(kwargs)
        raise AssertionError("dry-run reached T2 capture")

    monkeypatch.setattr(t2, "t2_stream_metric_identifiers", lambda: ("metric",))
    monkeypatch.setattr(t2, "assert_targets_probe_available", lambda metrics: None)
    monkeypatch.setattr(t2, "capture_and_land", fail_if_called)
    output_paths = [
        t2.REPO_ROOT / "data/local_labels/test-t2-dry-run-labels.json",
        t2.REPO_ROOT / "data/local_labels/test-t2-dry-run-exclusions.json",
        t2.REPO_ROOT / "data/local_labels/test-t2-dry-run-governance.json",
        t2.REPO_ROOT / "data/local_labels/test-t2-dry-run-audit.json",
    ]
    result = t2.main(
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
    assert capsys.readouterr().err == (
        "[capture_t2_stream] --dry-run: no files written\n"
    )
    assert all(not path.exists() for path in output_paths)


def test_t2_dry_run_preserves_probe_gate(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def reject_probe(metrics: tuple[str, ...]) -> None:
        calls.append(metrics)
        raise RuntimeError("probe gate rejected T2 metrics")

    monkeypatch.setattr(t2, "t2_stream_metric_identifiers", lambda: ("metric",))
    monkeypatch.setattr(t2, "assert_targets_probe_available", reject_probe)
    monkeypatch.setattr(
        t2,
        "capture_and_land",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run skipped the probe gate")
        ),
    )
    with pytest.raises(RuntimeError, match="probe gate rejected"):
        t2.main(
            [
                "--label-admitted",
                str(t2.REPO_ROOT / "data/local_labels/test-t2-probe-labels.json"),
                "--exclusions-out",
                str(t2.REPO_ROOT / "data/local_labels/test-t2-probe-exclusions.json"),
                "--governance-out",
                str(t2.REPO_ROOT / "data/local_labels/test-t2-probe-governance.json"),
                "--audit-out",
                str(t2.REPO_ROOT / "data/local_labels/test-t2-probe-audit.json"),
                "--dry-run",
            ]
        )
    assert calls == [("metric",)]


def test_t2_dry_run_rejects_output_escape_before_capture(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        t2,
        "capture_and_land",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run accepted an unscoped output path")
        ),
    )
    with pytest.raises(ValueError, match="count_label_output_outside_quarantine"):
        t2.main(
            [
                "--label-admitted",
                str(tmp_path / "outside-labels.json"),
                "--dry-run",
            ]
        )


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
