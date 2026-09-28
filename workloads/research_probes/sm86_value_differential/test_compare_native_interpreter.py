from pathlib import Path

from compare_native_interpreter import compare


ROOT = Path(__file__).resolve().parent


def test_native_capture_matches_finite_interpreter() -> None:
    result = compare(
        ROOT / "value.sm_86.sass.txt",
        ROOT / "native_capture_delta_2pow31_seed17.out",
    )
    assert result["status"] == "ok"
    assert result["lanes_checked"] == 32
    assert result["claims"]["native_value_semantics_match"] is True
    assert result["claims"]["timing"] is False
