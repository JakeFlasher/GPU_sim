"""Tests for the repository CUTLASS path contract."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from tools.sol_baselines_harness import native_env


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_CUTLASS = REPO_ROOT / "third_party" / "cutlass"
BUILD_TEMPLATE = (
    REPO_ROOT
    / "tools"
    / "sol_execbench"
    / "src"
    / "sol_execbench"
    / "driver"
    / "templates"
    / "build_ext.py"
)


def _load_subprocess_env_module():
    """Load the migrated module without importing its stale package alias."""
    module_name = "_cutlass_path_contract_subprocess_env"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT
        / "tools"
        / "sol_baselines_harness"
        / "surfaces"
        / "subprocess_env.py",
    )
    assert spec is not None and spec.loader is not None

    package = types.ModuleType("baselines_harness")
    package.__path__ = []
    native_alias = types.ModuleType("baselines_harness.native_env")
    native_alias.default_cutlass_dir = native_env.default_cutlass_dir
    previous = {
        name: sys.modules.get(name)
        for name in ("baselines_harness", "baselines_harness.native_env")
    }
    sys.modules["baselines_harness"] = package
    sys.modules["baselines_harness.native_env"] = native_alias
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_template_validator():
    """Extract only the build template's pure CUTLASS validator."""
    tree = ast.parse(BUILD_TEMPLATE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validated_cutlass_dir"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Path": Path, "os": os}
    exec(compile(module, str(BUILD_TEMPLATE), "exec"), namespace)
    return namespace["_validated_cutlass_dir"]


def test_default_cutlass_dir_is_the_tracked_v461_tree() -> None:
    assert native_env.CUTLASS_DIR_DEFAULT == str(TRACKED_CUTLASS)
    assert native_env.default_cutlass_dir() == TRACKED_CUTLASS
    assert native_env.default_cutlass_dir() != REPO_ROOT / "cutlass"
    assert (
        TRACKED_CUTLASS / "include" / "cutlass" / "version.h"
    ).is_file()


def test_subprocess_build_env_injects_the_same_default() -> None:
    subprocess_env = _load_subprocess_env_module()

    built = subprocess_env.build_subprocess_env(
        source_env={"PATH": "/usr/bin"},
        extra_allowed=subprocess_env.BUILD_TOOLCHAIN_ALLOWED,
    )

    assert built["CUTLASS_DIR"] == str(TRACKED_CUTLASS)

    without_toolchain = subprocess_env.build_subprocess_env(
        source_env={"PATH": "/usr/bin"},
    )
    assert "CUTLASS_DIR" not in without_toolchain


def test_explicit_cutlass_dir_override_wins(tmp_path: Path) -> None:
    subprocess_env = _load_subprocess_env_module()
    override = tmp_path / "operator-cutlass"

    built = subprocess_env.build_subprocess_env(
        plan_env_overrides={"CUTLASS_DIR": str(override)},
        source_env={"PATH": "/usr/bin"},
        extra_allowed=subprocess_env.BUILD_TOOLCHAIN_ALLOWED,
    )

    assert built["CUTLASS_DIR"] == str(override)


def test_build_template_rejects_unset_cutlass_dir(monkeypatch) -> None:
    validate = _load_template_validator()
    monkeypatch.delenv("CUTLASS_DIR", raising=False)

    with pytest.raises(RuntimeError, match="CUTLASS_DIR is required"):
        validate()


def test_build_template_rejects_unusable_cutlass_dir(
    tmp_path: Path, monkeypatch,
) -> None:
    validate = _load_template_validator()
    monkeypatch.setenv("CUTLASS_DIR", str(tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="usable pinned tree"):
        validate()


def test_build_template_accepts_a_complete_explicit_tree(
    tmp_path: Path, monkeypatch,
) -> None:
    validate = _load_template_validator()
    cutlass = tmp_path / "operator-cutlass"
    (cutlass / "include" / "cutlass").mkdir(parents=True)
    (cutlass / "tools" / "util" / "include").mkdir(parents=True)
    (cutlass / "include" / "cutlass" / "version.h").write_text(
        "#define CUTLASS_MAJOR 4\n", encoding="utf-8",
    )
    monkeypatch.setenv("CUTLASS_DIR", str(cutlass))

    assert validate() == cutlass.resolve()
