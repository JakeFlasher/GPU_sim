import json
from pathlib import Path

from ptx_subset_compile_manifest import run


ROOT = Path(__file__).resolve().parents[1]


def test_bounded_compiler_manifest_binds_source_and_output(tmp_path: Path) -> None:
    result = run(
        ROOT / "sim/target/debug/ptx_subset_compile",
        "example_its",
        ROOT / "arch_profiles/programs/simt_its_subset.ptx",
        tmp_path / "program.json",
        tmp_path / "manifest.json",
        100_000,
        None,
    )
    assert result["schema"] == "ptx_subset_compiler_artifact_v1"
    assert result["contract"] == "ptx_subset_v2"
    assert result["claims"]["arbitrary_ptx"] is False
    assert result["claims"]["arbitrary_isa"] is False
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["generated_program"]["size_bytes"] == (tmp_path / "program.json").stat().st_size
