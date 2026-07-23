from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from tools.nvbit_tracer.runtime_env import TracerRuntimeError, tracer_runtime_env


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    tracer = tmp_path / "tracer.so"
    tracer.write_bytes(b"ELF fixture")
    pb_lib = tmp_path / "pb" / "lib"
    pb_lib.mkdir(parents=True)
    return tracer, pb_lib


def test_runtime_env_prepends_pinned_lib_once(tmp_path: Path) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    completed = subprocess.CompletedProcess(
        ["ldd"],
        0,
        f"libprotobuf.so.34.1.0 => {pb_lib}/libprotobuf.so.34.1.0 (0x1)\n"
        f"libabsl_base.so => {pb_lib}/libabsl_base.so (0x2)\n",
        "",
    )
    with mock.patch("tools.nvbit_tracer.runtime_env.subprocess.run", return_value=completed):
        env = tracer_runtime_env(
            tracer,
            pb_lib=pb_lib,
            base_env={"LD_LIBRARY_PATH": f"/other:{pb_lib}:/tail"},
        )
    assert env["LD_LIBRARY_PATH"] == f"{pb_lib}:/other:/tail"


def test_runtime_env_rejects_missing_pinned_runtime(tmp_path: Path) -> None:
    tracer = tmp_path / "tracer.so"
    tracer.write_bytes(b"ELF fixture")
    with pytest.raises(TracerRuntimeError, match="runtime directory is missing"):
        tracer_runtime_env(tracer, pb_lib=tmp_path / "absent", base_env={})


def test_runtime_env_rejects_unresolved_dependency(tmp_path: Path) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    completed = subprocess.CompletedProcess(
        ["ldd"], 0, "libprotobuf.so.34.1.0 => not found\n", ""
    )
    with mock.patch("tools.nvbit_tracer.runtime_env.subprocess.run", return_value=completed):
        with pytest.raises(TracerRuntimeError, match="silently run untraced"):
            tracer_runtime_env(tracer, pb_lib=pb_lib, base_env={})


def test_runtime_env_rejects_protobuf_from_wrong_prefix(tmp_path: Path) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    system_lib = tmp_path / "system" / "lib"
    system_lib.mkdir(parents=True)
    completed = subprocess.CompletedProcess(
        ["ldd"],
        0,
        f"libprotobuf.so.34.1.0 => {system_lib}/libprotobuf.so.34.1.0 (0x1)\n"
        f"libabsl_base.so => {pb_lib}/libabsl_base.so (0x2)\n",
        "",
    )
    with mock.patch("tools.nvbit_tracer.runtime_env.subprocess.run", return_value=completed):
        with pytest.raises(TracerRuntimeError, match="did not resolve from the pinned"):
            tracer_runtime_env(tracer, pb_lib=pb_lib, base_env={})


def test_runtime_env_uses_absolute_prefix_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    monkeypatch.setenv("NVBIT_PB_PREFIX", str(pb_lib.parent))
    completed = subprocess.CompletedProcess(
        ["ldd"], 0,
        f"libprotobuf.so.34.1.0 => {pb_lib}/libprotobuf.so.34.1.0 (0x1)\n",
        "",
    )
    with mock.patch(
        "tools.nvbit_tracer.runtime_env.subprocess.run", return_value=completed,
    ):
        env = tracer_runtime_env(tracer, base_env={})
    assert env["LD_LIBRARY_PATH"] == str(pb_lib)


def test_runtime_env_rejects_relative_or_symlinked_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    with pytest.raises(TracerRuntimeError, match="runtime path must be absolute"):
        tracer_runtime_env(tracer, pb_lib=Path("relative/pb/lib"), base_env={})
    monkeypatch.setenv("NVBIT_PB_PREFIX", "relative/pb")
    with pytest.raises(TracerRuntimeError, match="must be absolute"):
        tracer_runtime_env(tracer, base_env={})
    monkeypatch.delenv("NVBIT_PB_PREFIX")
    alias = tmp_path / "pb-link"
    alias.symlink_to(pb_lib.parent, target_is_directory=True)
    with pytest.raises(TracerRuntimeError, match="must not be a symlink"):
        tracer_runtime_env(tracer, pb_lib=alias / "lib", base_env={})


def test_pinned_environment_lock_and_interface_are_exact() -> None:
    lock = REPO_ROOT / "tools/nvbit_tracer/toolchain/pb_explicit.lock"
    payload = lock.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "d522b3ee1560e98512cd460b213adee5a901f77aef0e4fae3ba94699e3b1e222"
    )
    assert payload.decode("utf-8").splitlines()[0] == "@EXPLICIT"
    assert len(payload.decode("utf-8").splitlines()) == 8
    script = (
        REPO_ROOT / "tools/nvbit_tracer/pinned_build_env.sh"
    ).read_text(encoding="utf-8")
    assert 'pb_prefix="${NVBIT_PB_PREFIX:-$default_prefix}"' in script
    assert "--ensure may create only the canonical .toolchain/pb prefix" in script
    assert "never repairs or replaces a present prefix" in script


def test_pinned_environment_lock_and_missing_prefix_exit_taxonomy(
    tmp_path: Path,
) -> None:
    source_script = REPO_ROOT / "tools/nvbit_tracer/pinned_build_env.sh"
    source_lock = REPO_ROOT / "tools/nvbit_tracer/toolchain/pb_explicit.lock"
    script = tmp_path / "repo/tools/nvbit_tracer/pinned_build_env.sh"
    lock = script.parent / "toolchain/pb_explicit.lock"
    lock.parent.mkdir(parents=True)
    script.write_bytes(source_script.read_bytes())
    env = os.environ.copy()
    env.pop("NVBIT_PB_PREFIX", None)

    missing = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True,
    )
    assert missing.returncode == 2
    assert "exact protobuf lock is missing" in missing.stderr

    lock.write_bytes(source_lock.read_bytes() + b"#corrupt\n")
    mismatched = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True,
    )
    assert mismatched.returncode == 1
    assert "protobuf lock mismatch" in mismatched.stderr

    lock.write_bytes(source_lock.read_bytes())
    absent_prefix = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True,
    )
    assert absent_prefix.returncode == 2
    assert "pinned protobuf prefix is missing" in absent_prefix.stderr
