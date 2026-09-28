import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_retained_native_message_passing_litmus() -> None:
    line = next(line for line in (ROOT / "native_litmus_result_1000.out").read_text().splitlines() if line.startswith("LITMUS "))
    result = json.loads(line[len("LITMUS ") :])
    assert result == {
        "schema": "sm86_message_passing_litmus_v1",
        "iterations": 1000,
        "failures": 0,
        "timeouts": 0,
        "status": "ok",
    }
    manifest = json.loads((ROOT / "native_litmus_manifest.json").read_text())
    assert manifest["claims"]["hardware_value_observation"] is True
    assert manifest["claims"]["simulator_coherence_completion"] is False


def test_retained_l1_litmus_separates_visibility_controls() -> None:
    line = next(line for line in (ROOT / "native_l1_invalidation_result_1000.out").read_text().splitlines() if line.startswith("LITMUS "))
    result = json.loads(line[len("LITMUS ") :])
    assert result["schema"] == "sm86_l1_invalidation_litmus_v1"
    assert result["iterations"] == 1000
    assert result["failures"] > 0
    assert result["volatile_failures"] == 0
    assert result["consumer_fence_failures"] == 0
