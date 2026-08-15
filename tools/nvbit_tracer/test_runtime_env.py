from __future__ import annotations

import ast
import errno
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.nvbit_tracer import runtime_env as runtime_env_module
from tools.nvbit_tracer.runtime_env import (
    NVBIT_CAPTURE_ENVIRONMENT_CONTROLS,
    TracerRuntimeError,
    retained_tracer_runtime_env,
    sanitized_capture_environment,
    tracer_runtime_env,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_runtime_manifest(
    tracer: Path,
    pb_lib: Path,
    dsos: dict[str, bytes],
    *,
    manifest_path: Path | None = None,
    write_dsos: bool = True,
) -> Path:
    rows = []
    for soname, payload in sorted(dsos.items(), key=lambda item: os.fsencode(item[0])):
        dso_path = pb_lib / soname
        if write_dsos:
            dso_path.write_bytes(payload)
        rows.append({
            "path": soname,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "soname": soname,
        })
    document = immutable.seal_model({
        "authority_kind": runtime_env_module.RUNTIME_MANIFEST_AUTHORITY_KIND,
        "dsos": rows,
        "runtime_library_path": str(pb_lib),
        "schema_version": runtime_env_module.RUNTIME_MANIFEST_SCHEMA_VERSION,
        "tracer": {
            "sha256": hashlib.sha256(tracer.read_bytes()).hexdigest(),
            "size_bytes": tracer.stat().st_size,
        },
    })
    selected = (
        runtime_env_module.default_runtime_manifest_path(tracer)
        if manifest_path is None
        else manifest_path
    )
    selected.write_bytes(immutable.pretty_json_bytes(document))
    return selected


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    tracer = tmp_path / "tracer.so"
    tracer.write_bytes(b"ELF fixture")
    pb_lib = tmp_path / "pb" / "lib"
    pb_lib.mkdir(parents=True)
    _write_runtime_manifest(
        tracer,
        pb_lib,
        {"libprotobuf.so.34.1.0": b"fixture protobuf DSO"},
    )
    return tracer, pb_lib


def test_capture_environment_removes_loader_and_every_tracer_control() -> None:
    hostile = {name: f"hostile-{name}" for name in NVBIT_CAPTURE_ENVIRONMENT_CONTROLS}
    hostile["PATH"] = "/trusted/path"

    assert sanitized_capture_environment(hostile) == {"PATH": "/trusted/path"}


def test_runtime_env_replaces_ambient_loader_path(tmp_path: Path) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    completed = subprocess.CompletedProcess(
        ["ldd"],
        0,
        f"libprotobuf.so.34.1.0 => {pb_lib}/libprotobuf.so.34.1.0 (0x1)\n"
        f"libabsl_base.so => {pb_lib}/libabsl_base.so (0x2)\n",
        "",
    )
    with mock.patch("tools.nvbit_tracer.runtime_env._run_ldd", return_value=completed):
        env = tracer_runtime_env(
            tracer,
            pb_lib=pb_lib,
            base_env={"LD_LIBRARY_PATH": f"/other:{pb_lib}:/tail"},
        )
    assert env["LD_LIBRARY_PATH"] == str(pb_lib)


def test_runtime_env_preflights_retained_tracer_descriptor(tmp_path: Path) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    completed = subprocess.CompletedProcess(
        ["ldd"],
        0,
        f"libprotobuf.so.34.1.0 => {pb_lib}/libprotobuf.so.34.1.0 (0x1)\n",
        "",
    )
    descriptor = os.open(tracer, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            return_value=completed,
        ) as run:
            tracer_runtime_env(
                tracer,
                pb_lib=pb_lib,
                base_env={},
                retained_descriptor=descriptor,
            )
    finally:
        os.close(descriptor)

    args, kwargs = run.call_args
    assert args[0] == f"/proc/self/fd/{descriptor}"
    assert kwargs["pass_fds"] == (descriptor,)


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
    with mock.patch("tools.nvbit_tracer.runtime_env._run_ldd", return_value=completed):
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
    with mock.patch("tools.nvbit_tracer.runtime_env._run_ldd", return_value=completed):
        with pytest.raises(TracerRuntimeError, match="did not resolve from the pinned"):
            tracer_runtime_env(tracer, pb_lib=pb_lib, base_env={})


def test_runtime_env_ignores_ambient_prefix_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    hostile_lib = tmp_path / "hostile" / "lib"
    hostile_lib.mkdir(parents=True)
    monkeypatch.setenv("NVBIT_PB_PREFIX", str(hostile_lib.parent))
    monkeypatch.setattr(runtime_env_module, "DEFAULT_PB_LIB", pb_lib)
    completed = subprocess.CompletedProcess(
        ["ldd"], 0,
        f"libprotobuf.so.34.1.0 => {pb_lib}/libprotobuf.so.34.1.0 (0x1)\n",
        "",
    )
    with mock.patch(
        "tools.nvbit_tracer.runtime_env._run_ldd", return_value=completed,
    ):
        env = tracer_runtime_env(tracer)
    assert env["LD_LIBRARY_PATH"] == str(pb_lib)
    assert "NVBIT_PB_PREFIX" not in env


def test_runtime_env_explicit_base_ignores_hostile_ambient_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, trusted_lib = _fixture_paths(tmp_path)
    hostile_lib = tmp_path / "hostile" / "lib"
    hostile_lib.mkdir(parents=True)
    monkeypatch.setenv("NVBIT_PB_PREFIX", str(hostile_lib.parent))
    monkeypatch.setattr(runtime_env_module, "DEFAULT_PB_LIB", trusted_lib)
    completed = subprocess.CompletedProcess(
        ["ldd"], 0,
        f"libprotobuf.so.34.1.0 => {trusted_lib}/libprotobuf.so.34.1.0 (0x1)\n",
        "",
    )

    with mock.patch(
        "tools.nvbit_tracer.runtime_env._run_ldd", return_value=completed,
    ):
        env = tracer_runtime_env(tracer, base_env={})

    assert env["LD_LIBRARY_PATH"] == str(trusted_lib)
    assert "NVBIT_PB_PREFIX" not in env


def test_runtime_env_omitted_base_still_ignores_ambient_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    hostile_lib = tmp_path / "hostile" / "lib"
    hostile_lib.mkdir(parents=True)
    monkeypatch.setenv("NVBIT_PB_PREFIX", str(hostile_lib.parent))
    monkeypatch.setattr(runtime_env_module, "DEFAULT_PB_LIB", pb_lib)
    completed = subprocess.CompletedProcess(
        ["ldd"], 0,
        f"libprotobuf.so.34.1.0 => {pb_lib}/libprotobuf.so.34.1.0 (0x1)\n",
        "",
    )

    with mock.patch(
        "tools.nvbit_tracer.runtime_env._run_ldd", return_value=completed,
    ):
        env = tracer_runtime_env(tracer, base_env=None)

    assert env["LD_LIBRARY_PATH"] == str(pb_lib)
    assert "NVBIT_PB_PREFIX" not in env


def test_runtime_env_rejects_relative_or_symlinked_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    with pytest.raises(TracerRuntimeError, match="runtime path must be absolute"):
        tracer_runtime_env(tracer, pb_lib=Path("relative/pb/lib"), base_env={})
    monkeypatch.setenv("NVBIT_PB_PREFIX", "relative/pb")
    monkeypatch.setattr(runtime_env_module, "DEFAULT_PB_LIB", pb_lib)
    completed = subprocess.CompletedProcess(
        ["ldd"], 0,
        f"libprotobuf.so.34.1.0 => {pb_lib}/libprotobuf.so.34.1.0 (0x1)\n",
        "",
    )
    with mock.patch(
        "tools.nvbit_tracer.runtime_env._run_ldd", return_value=completed,
    ):
        assert tracer_runtime_env(tracer)["LD_LIBRARY_PATH"] == str(pb_lib)
    monkeypatch.delenv("NVBIT_PB_PREFIX")
    alias = tmp_path / "pb-link"
    alias.symlink_to(pb_lib.parent, target_is_directory=True)
    with pytest.raises(TracerRuntimeError, match="must not be a symlink"):
        tracer_runtime_env(tracer, pb_lib=alias / "lib", base_env={})


def _retained_runtime_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, bytes]]:
    tracer, pb_lib = _fixture_paths(tmp_path)
    dsos = {
        "libprotobuf.so.34.1.0": b"authenticated protobuf DSO bytes",
        "libabsl_base.so.2407": b"authenticated Abseil DSO bytes",
    }
    _write_runtime_manifest(tracer, pb_lib, dsos)
    return tracer, pb_lib, dsos


def _descriptor_ldd_result(
    tracer: Path,
    *,
    dsos: dict[str, bytes],
):
    def run(tracer_path, *, env, pass_fds):
        tracer_descriptor = int(tracer_path.rsplit("/", 1)[1])
        assert tracer_descriptor in pass_fds
        runtime_entry = env["LD_LIBRARY_PATH"].split(":", 1)[0]
        runtime_descriptor = int(runtime_entry.rsplit("/", 1)[1])
        assert runtime_descriptor in pass_fds
        runtime_path = Path(os.readlink(runtime_entry))
        return subprocess.CompletedProcess(
            ["ldd", tracer_path],
            0,
            "".join(
                f"{soname} => {runtime_path / soname} (0x1)\n"
                for soname in dsos
            ),
            "",
        )

    return run


def test_retained_runtime_uses_private_hash_bound_descriptor_tree(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    descriptor = os.open(tracer, os.O_RDONLY)
    private_path: Path | None = None
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ) as run:
            with retained_tracer_runtime_env(
                tracer,
                pb_lib=pb_lib,
                base_env={"LD_LIBRARY_PATH": "/other"},
                retained_descriptor=descriptor,
            ) as runtime:
                assert run.call_count == 2
                runtime_entry = runtime.environment["LD_LIBRARY_PATH"].split(
                    ":", 1
                )[0]
                runtime_descriptor = int(runtime_entry.rsplit("/", 1)[1])
                assert runtime.pass_fds == (runtime_descriptor,)
                private_path = Path(os.readlink(runtime_entry))
                assert private_path != pb_lib
                assert private_path.stat().st_mode & 0o777 == 0o555
                assert runtime.environment["LD_LIBRARY_PATH"] == runtime_entry
                assert runtime.dso_sha256 == {
                    soname: hashlib.sha256(payload).hexdigest()
                    for soname, payload in dsos.items()
                }
                for soname, payload in dsos.items():
                    private_dso = private_path / soname
                    assert private_dso.read_bytes() == payload
                    assert private_dso.stat().st_mode & 0o777 == 0o444
                    assert private_dso.stat().st_ino != (pb_lib / soname).stat().st_ino
                runtime.verify_unchanged()
    finally:
        os.close(descriptor)
    assert private_path is not None
    assert not private_path.exists()


def test_retained_runtime_uses_one_sanitized_snapshot_for_both_ldd_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, trusted_lib, dsos = _retained_runtime_fixture(tmp_path)
    hostile_lib = tmp_path / "hostile" / "lib"
    hostile_lib.mkdir(parents=True)
    monkeypatch.setenv("NVBIT_PB_PREFIX", str(hostile_lib.parent))
    monkeypatch.setattr(runtime_env_module, "DEFAULT_PB_LIB", trusted_lib)
    supplied = sanitized_capture_environment({
        "NVBIT_PB_PREFIX": str(hostile_lib.parent),
        "FIXTURE": "trusted",
        "LD_LIBRARY_PATH": "/other",
    })
    descriptor = os.open(tracer, os.O_RDONLY)
    observed_environments: list[dict[str, str]] = []
    run_ldd = _descriptor_ldd_result(tracer, dsos=dsos)

    def recording_ldd(tracer_path, *, env, pass_fds):
        observed_environments.append(dict(env))
        return run_ldd(tracer_path, env=env, pass_fds=pass_fds)

    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=recording_ldd,
        ):
            with retained_tracer_runtime_env(
                tracer,
                base_env=supplied,
                retained_descriptor=descriptor,
            ) as runtime:
                assert runtime.environment["FIXTURE"] == "trusted"
                assert "NVBIT_PB_PREFIX" not in runtime.environment
                assert runtime.dso_sha256 == {
                    soname: hashlib.sha256(payload).hexdigest()
                    for soname, payload in dsos.items()
                }
    finally:
        os.close(descriptor)

    assert len(observed_environments) == 2
    assert all(
        environment["FIXTURE"] == "trusted"
        and "NVBIT_PB_PREFIX" not in environment
        for environment in observed_environments
    )
    assert all(
        not environment["LD_LIBRARY_PATH"].split(":", 1)[0].startswith(
            str(hostile_lib)
        )
        for environment in observed_environments
    )
    assert all(
        environment["LD_LIBRARY_PATH"].startswith("/proc/self/fd/")
        and ":" not in environment["LD_LIBRARY_PATH"]
        and "/other" not in environment["LD_LIBRARY_PATH"]
        for environment in observed_environments
    )


def test_retained_runtime_accepts_explicit_operator_manifest(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    runtime_env_module.default_runtime_manifest_path(tracer).unlink()
    explicit_manifest = _write_runtime_manifest(
        tracer,
        pb_lib,
        dsos,
        manifest_path=tmp_path / "operator-runtime-authority.json",
        write_dsos=False,
    )
    descriptor = os.open(tracer, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ):
            with retained_tracer_runtime_env(
                tracer,
                pb_lib=pb_lib,
                runtime_manifest=explicit_manifest,
                base_env={"LD_LIBRARY_PATH": "/ambient/injection"},
                retained_descriptor=descriptor,
            ) as runtime:
                assert runtime.runtime_manifest_path == explicit_manifest
                assert runtime.environment["LD_LIBRARY_PATH"].startswith(
                    "/proc/self/fd/"
                )
                assert ":" not in runtime.environment["LD_LIBRARY_PATH"]
    finally:
        os.close(descriptor)


def test_retained_runtime_binds_caller_retained_manifest_descriptor(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    manifest = runtime_env_module.default_runtime_manifest_path(tracer)
    tracer_descriptor = os.open(tracer, os.O_RDONLY)
    manifest_descriptor = os.open(manifest, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ):
            with retained_tracer_runtime_env(
                tracer,
                pb_lib=pb_lib,
                runtime_manifest=manifest,
                base_env={},
                retained_descriptor=tracer_descriptor,
                retained_manifest_descriptor=manifest_descriptor,
            ) as runtime:
                assert runtime.runtime_manifest_path == manifest
                assert runtime.runtime_manifest_sha256 == hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest()
    finally:
        os.close(manifest_descriptor)
        os.close(tracer_descriptor)


def test_retained_runtime_rejects_manifest_descriptor_for_different_inode(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, _dsos = _retained_runtime_fixture(tmp_path)
    manifest = runtime_env_module.default_runtime_manifest_path(tracer)
    alternate_manifest = tmp_path / "alternate-runtime-authority.json"
    alternate_manifest.write_bytes(manifest.read_bytes())
    tracer_descriptor = os.open(tracer, os.O_RDONLY)
    manifest_descriptor = os.open(alternate_manifest, os.O_RDONLY)
    try:
        with mock.patch("tools.nvbit_tracer.runtime_env._run_ldd") as run:
            with pytest.raises(TracerRuntimeError) as caught:
                retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    runtime_manifest=manifest,
                    base_env={},
                    retained_descriptor=tracer_descriptor,
                    retained_manifest_descriptor=manifest_descriptor,
                )
    finally:
        os.close(manifest_descriptor)
        os.close(tracer_descriptor)

    assert caught.value.reason == "runtime_authority_changed"
    assert "differs from the caller-retained descriptor" in caught.value.detail
    run.assert_not_called()


def test_retained_runtime_rejects_retained_manifest_byte_mutation(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    manifest = runtime_env_module.default_runtime_manifest_path(tracer)
    tracer_descriptor = os.open(tracer, os.O_RDONLY)
    manifest_descriptor = os.open(manifest, os.O_RDONLY)
    runtime = None
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ):
            runtime = retained_tracer_runtime_env(
                tracer,
                pb_lib=pb_lib,
                runtime_manifest=manifest,
                base_env={},
                retained_descriptor=tracer_descriptor,
                retained_manifest_descriptor=manifest_descriptor,
            )
            manifest.write_bytes(b"mutated retained runtime authority")
            with pytest.raises(TracerRuntimeError) as caught:
                runtime.verify_unchanged()
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except TracerRuntimeError as cleanup_error:
                assert cleanup_error.reason == "runtime_dso_generation_changed"
        os.close(manifest_descriptor)
        os.close(tracer_descriptor)

    assert caught.value.reason == "runtime_authority_changed"
    assert "retained runtime manifest changed" in caught.value.detail


def test_retained_runtime_missing_manifest_is_typed_prerequisite(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, _dsos = _retained_runtime_fixture(tmp_path)
    runtime_env_module.default_runtime_manifest_path(tracer).unlink()
    descriptor = os.open(tracer, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
        ) as run:
            with pytest.raises(TracerRuntimeError) as caught:
                retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    base_env={"LD_LIBRARY_PATH": "/hostile"},
                    retained_descriptor=descriptor,
                )
    finally:
        os.close(descriptor)

    assert caught.value.reason == "runtime_authority_prerequisite_missing"
    assert "independent build provenance" in caught.value.detail
    run.assert_not_called()


def test_retained_runtime_rejects_prefix_bytes_not_matching_manifest(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    (pb_lib / "libprotobuf.so.34.1.0").write_bytes(
        b"arbitrary replacement protobuf DSO bytes"
    )
    descriptor = os.open(tracer, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ):
            with pytest.raises(TracerRuntimeError) as caught:
                retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    base_env={},
                    retained_descriptor=descriptor,
                )
    finally:
        os.close(descriptor)

    assert caught.value.reason == "runtime_dso_authority_mismatch"
    assert "differs from independent runtime authority" in caught.value.detail


def test_retained_runtime_rejects_discovered_dso_absent_from_manifest(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    _write_runtime_manifest(
        tracer,
        pb_lib,
        {"libprotobuf.so.34.1.0": dsos["libprotobuf.so.34.1.0"]},
        write_dsos=False,
    )
    descriptor = os.open(tracer, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ):
            with pytest.raises(TracerRuntimeError) as caught:
                retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    base_env={},
                    retained_descriptor=descriptor,
                )
    finally:
        os.close(descriptor)

    assert caught.value.reason == "runtime_dso_authority_mismatch"
    assert "unknown=['libabsl_base.so.2407']" in caught.value.detail


def test_retained_runtime_rejects_manifest_for_different_tracer(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, _dsos = _retained_runtime_fixture(tmp_path)
    tracer.write_bytes(b"different tracer generation")
    descriptor = os.open(tracer, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
        ) as run:
            with pytest.raises(TracerRuntimeError) as caught:
                retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    base_env={},
                    retained_descriptor=descriptor,
                )
    finally:
        os.close(descriptor)

    assert caught.value.reason == "runtime_authority_tracer_mismatch"
    run.assert_not_called()


def test_retained_runtime_sanitizes_unsanitized_supplied_capture_controls(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    descriptor = os.open(tracer, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ):
            with retained_tracer_runtime_env(
                tracer,
                pb_lib=pb_lib,
                base_env={
                    "CUDA_INJECTION64_PATH": "/tmp/hostile-tracer.so",
                    "NVBIT_TOOL": "hostile",
                    "NVBIT_PB_PREFIX": "/tmp/hostile-prefix",
                    "PATH": "/tmp/hostile-bin",
                    "FIXTURE": "preserved",
                },
                retained_descriptor=descriptor,
            ) as runtime:
                assert runtime.environment["FIXTURE"] == "preserved"
                assert "CUDA_INJECTION64_PATH" not in runtime.environment
                assert "NVBIT_TOOL" not in runtime.environment
                assert "NVBIT_PB_PREFIX" not in runtime.environment
    finally:
        os.close(descriptor)


def test_ldd_executes_retained_capability_with_nonambient_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen
    observed: dict[str, object] = {}

    def replacement(command, **kwargs):
        observed["command"] = tuple(command)
        observed["environment"] = dict(kwargs["env"])
        observed["pass_fds"] = tuple(kwargs["pass_fds"])
        return real_popen(
            [sys.executable, "-c", "pass"],
            **kwargs,
        )

    monkeypatch.setattr(runtime_env_module.subprocess, "Popen", replacement)
    completed = runtime_env_module._run_ldd(
        "fixture.so",
        env={
            "PATH": "/tmp/hostile-bin",
            "BASH_ENV": "/tmp/hostile-bash-env",
            "ENV": "/tmp/hostile-env",
        },
        pass_fds=(),
    )
    assert completed.returncode == 0
    command = observed["command"]
    assert command[0].startswith("/proc/self/fd/")
    assert command[1] == "fixture.so"
    assert observed["environment"]["PATH"] == "/usr/bin:/bin"
    assert "BASH_ENV" not in observed["environment"]
    assert "ENV" not in observed["environment"]
    assert int(command[0].rsplit("/", 1)[1]) in observed["pass_fds"]


def test_retained_runtime_rejects_source_alias_replacement_during_authentication(
    tmp_path: Path,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    descriptor = os.open(tracer, os.O_RDONLY)
    original = __import__(
        "tools.nvbit_tracer.runtime_env", fromlist=["_private_runtime_tree"]
    )._private_runtime_tree

    def replace_alias(resources, targets):
        alias = pb_lib / "libprotobuf.so.34.1.0"
        displaced = pb_lib / "libprotobuf.so.displaced"
        alias.rename(displaced)
        alias.write_bytes(b"replacement DSO")
        return original(resources, targets)

    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ), mock.patch(
            "tools.nvbit_tracer.runtime_env._private_runtime_tree",
            side_effect=replace_alias,
        ):
            with pytest.raises(TracerRuntimeError) as caught:
                retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    base_env={},
                    retained_descriptor=descriptor,
                )
    finally:
        os.close(descriptor)
    assert caught.value.reason == "runtime_dso_generation_changed"


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (
            immutable.ImmutableArtifactLimitError("bounded inventory exceeded"),
            "runtime_dso_limit_exceeded",
        ),
        (MemoryError("allocation failed"), "runtime_resource_unavailable"),
    ],
)
def test_retained_runtime_preserves_limit_and_resource_failure_taxonomy(
    tmp_path: Path,
    failure: BaseException,
    reason: str,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    descriptor = os.open(tracer, os.O_RDONLY)
    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ), mock.patch(
            "tools.nvbit_tracer.runtime_env._private_runtime_tree",
            side_effect=failure,
        ):
            with pytest.raises(TracerRuntimeError) as caught:
                retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    base_env={},
                    retained_descriptor=descriptor,
                )
    finally:
        os.close(descriptor)
    assert caught.value.reason == reason


def test_ldd_timeout_is_bounded_typed_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def sleeping_ldd(_command, **kwargs):
        process = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(runtime_env_module.subprocess, "Popen", sleeping_ldd)
    monkeypatch.setattr(runtime_env_module, "LDD_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(TracerRuntimeError) as caught:
        runtime_env_module._run_ldd("fixture.so", env={}, pass_fds=())

    assert caught.value.reason == "ldd_timeout"
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout is not None and processes[0].stdout.closed
    assert processes[0].stderr is not None and processes[0].stderr.closed


@pytest.mark.parametrize(
    "failure",
    [
        MemoryError("synthetic ldd allocation failure"),
        OSError(errno.EAGAIN, "synthetic ldd process limit"),
        OSError(errno.EMFILE, "synthetic ldd descriptor limit"),
        OSError(errno.ENFILE, "synthetic system descriptor limit"),
        OSError(errno.ENOMEM, "synthetic ldd memory exhaustion"),
    ],
)
def test_ldd_start_resource_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    def fail_start(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(runtime_env_module.subprocess, "Popen", fail_start)
    with pytest.raises(TracerRuntimeError) as caught:
        runtime_env_module._run_ldd("fixture.so", env={}, pass_fds=())

    assert caught.value.reason == "runtime_resource_unavailable"


@pytest.mark.parametrize(
    "resource_errno", [errno.EAGAIN, errno.EMFILE, errno.ENFILE, errno.ENOMEM],
)
def test_ldd_selector_resource_failure_reaps_and_closes_pipes(
    monkeypatch: pytest.MonkeyPatch,
    resource_errno: int,
) -> None:
    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def sleeping_ldd(_command, **kwargs):
        process = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            **kwargs,
        )
        processes.append(process)
        return process

    def fail_selector_setup():
        raise OSError(resource_errno, "synthetic selector resource failure")

    monkeypatch.setattr(runtime_env_module.subprocess, "Popen", sleeping_ldd)
    monkeypatch.setattr(
        runtime_env_module.selectors, "DefaultSelector", fail_selector_setup,
    )
    with pytest.raises(TracerRuntimeError) as caught:
        runtime_env_module._run_ldd("fixture.so", env={}, pass_fds=())

    assert caught.value.reason == "runtime_resource_unavailable"
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout is not None and processes[0].stdout.closed
    assert processes[0].stderr is not None and processes[0].stderr.closed


def test_ldd_selector_setup_failure_is_typed_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def sleeping_ldd(_command, **kwargs):
        process = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            **kwargs,
        )
        processes.append(process)
        return process

    def fail_selector_setup():
        raise OSError(errno.EIO, "synthetic selector setup failure")

    monkeypatch.setattr(runtime_env_module.subprocess, "Popen", sleeping_ldd)
    monkeypatch.setattr(
        runtime_env_module.selectors,
        "DefaultSelector",
        fail_selector_setup,
    )
    with pytest.raises(TracerRuntimeError) as caught:
        runtime_env_module._run_ldd("fixture.so", env={}, pass_fds=())

    assert caught.value.reason == "ldd_output_read_failed"
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout is not None and processes[0].stdout.closed
    assert processes[0].stderr is not None and processes[0].stderr.closed


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_ldd_output_limit_is_typed_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
) -> None:
    stream_fd = 1 if stream_name == "stdout" else 2
    limit = (
        runtime_env_module.LDD_MAX_STDOUT_BYTES
        if stream_name == "stdout"
        else runtime_env_module.LDD_MAX_STDERR_BYTES
    )
    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def noisy_ldd(_command, **kwargs):
        process = real_popen(
            [
                sys.executable,
                "-c",
                "import os, time; "
                f"os.write({stream_fd}, b'x' * ({limit} + 1)); "
                "time.sleep(60)",
            ],
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(runtime_env_module.subprocess, "Popen", noisy_ldd)
    with pytest.raises(TracerRuntimeError) as caught:
        runtime_env_module._run_ldd("fixture.so", env={}, pass_fds=())

    assert caught.value.reason == "ldd_output_limit_exceeded"
    assert stream_name in caught.value.detail
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout is not None and processes[0].stdout.closed
    assert processes[0].stderr is not None and processes[0].stderr.closed


def test_failed_retained_runtime_context_entry_closes_private_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    descriptor = os.open(tracer, os.O_RDONLY)
    private_path: Path | None = None
    real_private_tree = runtime_env_module._private_runtime_tree
    real_verify = runtime_env_module.RetainedTracerRuntime.verify_unchanged
    verify_calls = 0

    def capture_private_tree(resources, targets):
        nonlocal private_path
        result = real_private_tree(resources, targets)
        private_path = result[0]
        return result

    def fail_context_entry(runtime):
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise TracerRuntimeError(
                "runtime_dso_generation_changed",
                "synthetic context entry identity failure",
            )
        return real_verify(runtime)

    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ), mock.patch(
            "tools.nvbit_tracer.runtime_env._private_runtime_tree",
            side_effect=capture_private_tree,
        ), mock.patch.object(
            runtime_env_module.RetainedTracerRuntime,
            "verify_unchanged",
            fail_context_entry,
        ):
            with pytest.raises(TracerRuntimeError) as caught:
                with retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    base_env={},
                    retained_descriptor=descriptor,
                ):
                    pytest.fail("context body ran after entry verification failure")
    finally:
        os.close(descriptor)

    assert caught.value.reason == "runtime_dso_generation_changed"
    assert verify_calls == 2
    assert private_path is not None
    assert not private_path.exists()


@pytest.mark.parametrize("open_target", ["directory", "dso"])
@pytest.mark.parametrize(
    ("error_number", "expected_reason"),
    [
        (errno.EMFILE, "runtime_resource_unavailable"),
        (errno.ENFILE, "runtime_resource_unavailable"),
        (errno.ENOMEM, "runtime_resource_unavailable"),
        (errno.EIO, "runtime_dso_generation_changed"),
    ],
)
def test_private_runtime_open_preserves_failure_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    expected_reason: str,
    open_target: str,
) -> None:
    tracer, pb_lib, dsos = _retained_runtime_fixture(tmp_path)
    descriptor = os.open(tracer, os.O_RDONLY)
    real_open = os.open
    failed_path: Path | None = None

    def fail_private_runtime_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal failed_path
        candidate = Path(os.fspath(path))
        if dir_fd is None and candidate.name.startswith("nvbit-runtime-"):
            failed_path = candidate
            if open_target == "directory":
                raise OSError(
                    error_number,
                    "synthetic private runtime directory open failure",
                )
        if (
            dir_fd is not None
            and open_target == "dso"
            and failed_path is not None
            and candidate.name in dsos
        ):
            raise OSError(
                error_number,
                "synthetic private runtime DSO open failure",
            )
        return real_open(path, flags, mode, dir_fd=dir_fd)

    try:
        with mock.patch(
            "tools.nvbit_tracer.runtime_env._run_ldd",
            side_effect=_descriptor_ldd_result(tracer, dsos=dsos),
        ):
            monkeypatch.setattr(
                runtime_env_module.os,
                "open",
                fail_private_runtime_open,
            )
            with pytest.raises(TracerRuntimeError) as caught:
                retained_tracer_runtime_env(
                    tracer,
                    pb_lib=pb_lib,
                    base_env={},
                    retained_descriptor=descriptor,
                )
    finally:
        os.close(descriptor)

    assert caught.value.reason == expected_reason
    assert failed_path is not None
    assert not failed_path.exists()


def test_active_split_tracer_consumers_retain_private_runtime_context() -> None:
    active_consumers = (
        (
            REPO_ROOT / "tools/local_labels/campaign_preflight.py",
            "probe_injection_runtime",
        ),
        (
            REPO_ROOT
            / "tools/local_labels/emit_long_timing_feature_artifacts.py",
            "_capture_nvbit_observed_sector_trace",
        ),
        (
            REPO_ROOT
            / "tools/local_labels/capture_cutlass_successor_receipts.py",
            "capture_phase",
        ),
        (
            REPO_ROOT
            / "tools/trace_sampling/golden_cells/retained_nvbit_launch.py",
            "main",
        ),
        (
            REPO_ROOT
            / "tools/local_labels/activation_trace_orchestrator.py",
            "trace_activation_cell",
        ),
        (
            REPO_ROOT / "tools/local_labels/roster_trace_capture.py",
            "_capture_direct",
        ),
        (
            REPO_ROOT
            / "tools/local_labels/trace_register_cap_variant_cells.py",
            "trace_variant_cell",
        ),
        (
            REPO_ROOT / "tools/local_labels/sm86_capture_chunk.py",
            "_authenticated_tracer_runtime_environment",
        ),
        (
            REPO_ROOT
            / "tools/local_labels/canary/emit_canary_manifest.py",
            "capture",
        ),
        (
            REPO_ROOT / "tools/trace_sampling/orchestrate.py",
            "_run_retained_pass",
        ),
        (
            REPO_ROOT / "workloads/simbench_bw/run_suite.py",
            "_run_retained_nvbit_capture",
        ),
        (
            REPO_ROOT
            / "workloads/simbench_bw/phase_inject_orchestrator.py",
            "run_ab_overhead_pair",
        ),
    )

    def called_helpers(node: ast.AST) -> set[str | None]:
        return {
            (
                call.func.id
                if isinstance(call.func, ast.Name)
                else call.func.attr
                if isinstance(call.func, ast.Attribute)
                else None
            )
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        }

    for path, function_name in active_consumers:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ]
        assert len(functions) == 1, (path, function_name)
        assert "retained_tracer_runtime_env" in called_helpers(functions[0]), (
            path,
            function_name,
        )
        runtime_calls = [
            node
            for node in ast.walk(functions[0])
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "retained_tracer_runtime_env"
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "retained_tracer_runtime_env"
                )
            )
        ]
        assert len(runtime_calls) == 1, (path, function_name)
        runtime_keywords = {
            keyword.arg: keyword.value for keyword in runtime_calls[0].keywords
        }
        assert "runtime_manifest" in runtime_keywords, (path, function_name)
        manifest_authority = runtime_keywords["runtime_manifest"]
        if path.name == "capture_cutlass_successor_receipts.py":
            assert isinstance(manifest_authority, ast.Name), (path, function_name)
            assert manifest_authority.id == "retained_runtime_manifest_path", (
                path,
                function_name,
            )
            retained_manifest_descriptor = runtime_keywords.get(
                "retained_manifest_descriptor"
            )
            assert isinstance(retained_manifest_descriptor, ast.Attribute), (
                path,
                function_name,
            )
            assert (
                retained_manifest_descriptor.attr == "descriptor"
                and isinstance(retained_manifest_descriptor.value, ast.Name)
                and retained_manifest_descriptor.value.id
                == "retained_runtime_manifest"
            ), (path, function_name)
        else:
            assert isinstance(manifest_authority, ast.Call), (path, function_name)
            assert (
                isinstance(manifest_authority.func, ast.Name)
                and manifest_authority.func.id == "default_runtime_manifest_path"
            ) or (
                isinstance(manifest_authority.func, ast.Attribute)
                and manifest_authority.func.attr == "default_runtime_manifest_path"
            ), (path, function_name)
        assert "tracer_runtime_env" not in called_helpers(tree), path


def test_production_capture_callers_pin_default_pb_lib_after_sanitizing() -> None:
    callers = (
        (
            REPO_ROOT / "tools/local_labels/activation_trace_orchestrator.py",
            "trace_activation_cell",
            0,
        ),
        (
            REPO_ROOT / "tools/local_labels/roster_trace_capture.py",
            "_capture_direct",
            0,
        ),
        (
            REPO_ROOT / "tools/local_labels/trace_register_cap_variant_cells.py",
            "trace_variant_cell",
            0,
        ),
        (
            REPO_ROOT / "workloads/simbench_bw/phase_inject_orchestrator.py",
            "run_ab_overhead_pair",
            1,
        ),
    )

    for path, function_name, sanitizer_argument_count in callers:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        calls = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "retained_tracer_runtime_env"
        ]
        assert len(calls) == 1, (path, function_name)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        assert isinstance(keywords["pb_lib"], ast.Name)
        assert keywords["pb_lib"].id == "DEFAULT_PB_LIB"
        sanitizer = keywords["base_env"]
        assert isinstance(sanitizer, ast.Call)
        assert isinstance(sanitizer.func, ast.Name)
        assert sanitizer.func.id == "sanitized_capture_environment"
        assert len(sanitizer.args) == sanitizer_argument_count


def test_cli_reports_missing_operator_manifest_prerequisite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracer, pb_lib = _fixture_paths(tmp_path)
    runtime_env_module.default_runtime_manifest_path(tracer).unlink()

    result = runtime_env_module.main([
        str(tracer), "--pb-lib", str(pb_lib),
    ])

    assert result == 1
    stderr = capsys.readouterr().err
    assert "runtime_authority_prerequisite_missing" in stderr
    assert "runtime discovery never mints these expected identities" in stderr


def test_pinned_environment_lock_shape_and_interface_are_explicit() -> None:
    lock = REPO_ROOT / "tools/nvbit_tracer/toolchain/pb_explicit.lock"
    payload = lock.read_bytes()
    lines = payload.decode("utf-8").splitlines()
    assert lines[0] == "@EXPLICIT"
    assert len(lines) == 8
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
    malformed = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True,
    )
    assert malformed.returncode == 1
    assert "protobuf package list is malformed" in malformed.stderr

    lock.write_bytes(source_lock.read_bytes())
    absent_prefix = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True,
    )
    assert absent_prefix.returncode == 2
    assert "pinned protobuf prefix is missing" in absent_prefix.stderr
