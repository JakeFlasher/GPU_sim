import json
from pathlib import Path

from native_capture_to_payload import convert


ROOT = Path(__file__).resolve().parent


def test_native_capture_converts_to_known_store_payloads() -> None:
    result = convert(ROOT / "native_capture_delta_2pow31_seed17.out")
    assert result["schema"] == "native_value_payload_sidecar_v1"
    assert result["status"] == "ok"
    assert len(result["observations"]) == 32
    assert len(result["payloads"]) == 32
    assert all(payload["allocation"] == "output" for payload in result["payloads"])
    assert all(payload["generation"] == 1 for payload in result["payloads"])
    assert all(payload["known_mask"] == (0xF << payload["byte_offset"]) for payload in result["payloads"])
    assert result["contract"]["coherence_ownership"] is False
    json.dumps(result)
