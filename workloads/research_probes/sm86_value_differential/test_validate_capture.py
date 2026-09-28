from pathlib import Path
import hashlib
import json

from validate_capture import validate


ROOT = Path(__file__).resolve().parent


def test_retained_native_sm86_capture_is_functionally_differential() -> None:
    result = validate(ROOT / "native_capture_delta_2pow31_seed17.out")
    assert result["status"] == "ok"
    assert result["functional_differential"] is True
    assert result["lanes_checked"] == 32
    assert result["predicate_true_lanes"] == 15
    assert result["claims"]["timing_validation"] is False
    assert result["claims"]["physical_pipe_reconstruction"] is False


def test_native_artifact_manifest_binds_retained_files() -> None:
    manifest = json.loads((ROOT / "native_artifact_manifest.json").read_text())
    assert manifest["schema"] == "sm86_value_differential_artifact_v1"
    for name, identity in manifest["artifacts"].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == identity["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == identity["sha256"]
