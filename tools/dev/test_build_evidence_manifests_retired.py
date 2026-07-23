from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "tools/dev/build_evidence_manifests.sh"
RETIREMENT_REASON = (
    "EVIDENCE_MANIFEST_BUILDER_RETIRED_POINT_IN_TIME_AUTHORITIES"
)


def test_point_in_time_manifest_builder_is_fail_closed() -> None:
    authorities = (
        REPO_ROOT / "evidence/manifests/external_capture.sha256",
        REPO_ROOT / "evidence/MANIFEST.sha256",
    )
    before = {path: path.read_bytes() for path in authorities}
    result = subprocess.run(
        [BUILDER],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert RETIREMENT_REASON in result.stderr
    assert "must not be overwritten or refreshed in place" in result.stderr
    assert {path: path.read_bytes() for path in authorities} == before


def test_external_capture_manifest_is_declared_point_in_time() -> None:
    manifest = REPO_ROOT / "evidence/manifests/external_capture.sha256"
    readme = REPO_ROOT / "evidence/manifests/README.md"

    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1135
    documentation = readme.read_text(encoding="utf-8")
    assert "1,135-file point-in-time snapshot" in documentation
    assert "must not be regenerated over" in documentation
