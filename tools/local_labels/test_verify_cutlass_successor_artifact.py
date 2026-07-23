from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.local_labels import verify_cutlass_successor_artifact as guard


REPO = Path(__file__).resolve().parents[2]
WORKLOAD = REPO / "workloads/simbench_bw/l1/vendor_gemm_cutlass"


def test_project_sources_pass_without_treating_explanatory_comments_as_imports():
    result = guard.verify_source_inputs(
        [WORKLOAD / "Makefile", WORKLOAD / "vendor_gemm_cutlass.cu", WORKLOAD / "main.cu"],
        repo_root=REPO,
    )
    assert set(result["files"]) == {
        "workloads/simbench_bw/l1/vendor_gemm_cutlass/Makefile",
        "workloads/simbench_bw/l1/vendor_gemm_cutlass/main.cu",
        "workloads/simbench_bw/l1/vendor_gemm_cutlass/vendor_gemm_cutlass.cu",
    }
    assert len(result["input_set_sha256"]) == 64


@pytest.mark.parametrize(
    "payload",
    [
        "#include <cublas_v2.h>\n",
        "return cublasGemmEx();\n",
        "NVCC_LIBS += -lcublasLt\n",
        'dlopen("libcublas.so.13", 0);\n',
    ],
)
def test_source_guard_rejects_forbidden_references(tmp_path: Path, payload: str):
    source = tmp_path / "bad.cu"
    source.write_text(payload)
    with pytest.raises(guard.CutlassArtifactError, match="forbidden cuBLAS"):
        guard.verify_source_inputs([source], repo_root=tmp_path)


def test_source_guard_rejects_symlink(tmp_path: Path):
    real = tmp_path / "real.cu"
    real.write_text("int main() {}\n")
    link = tmp_path / "link.cu"
    link.symlink_to(real)
    with pytest.raises(guard.CutlassArtifactError, match="regular file"):
        guard.verify_source_inputs([link], repo_root=tmp_path)


def test_elf_guard_accepts_clean_fixture(monkeypatch, tmp_path: Path):
    binary = tmp_path / "app"
    binary.write_bytes(b"fixture")
    binary.chmod(0o755)
    dynamic = b"Dynamic section at offset\n  NEEDED Shared library: [libcudart.so.13]\n"
    symbols = b"Symbol table '\\.dynsym' contains 1 entry:\n"
    monkeypatch.setattr(
        guard,
        "_run_readelf",
        lambda _binary, arguments: (dynamic if arguments[0] == "--dynamic" else symbols, b""),
    )
    result = guard.verify_elf_no_cublas(binary)
    assert result["needed_libraries"] == ["libcudart.so.13"]
    assert result["forbidden_dependency_count"] == 0


@pytest.mark.parametrize(
    "dynamic,symbols,needle",
    [
        (b"Shared library: [libcublas.so.13]\n", b"", "dependency"),
        (b"Shared library: [libcudart.so.13]\n", b"cublasGemmEx\n", "symbol"),
    ],
)
def test_elf_guard_rejects_forbidden_dependency_or_symbol(
    monkeypatch, tmp_path: Path, dynamic: bytes, symbols: bytes, needle: str,
):
    binary = tmp_path / "app"
    binary.write_bytes(b"fixture")
    binary.chmod(0o755)
    monkeypatch.setattr(
        guard,
        "_run_readelf",
        lambda _binary, arguments: (dynamic if arguments[0] == "--dynamic" else symbols, b""),
    )
    with pytest.raises(guard.CutlassArtifactError, match="forbidden cuBLAS"):
        guard.verify_elf_no_cublas(binary)


def test_current_binary_has_no_cublas_when_build_witness_exists():
    binary = WORKLOAD / "vendor_gemm_cutlass_run"
    if not binary.exists():
        pytest.skip("CUDA successor binary is not present in this checkout")
    result = guard.verify_elf_no_cublas(binary)
    assert result["needed_libraries"] == [
        "libc.so.6", "libcudart.so.13", "libgcc_s.so.1", "libm.so.6",
        "libstdc++.so.6",
    ]
