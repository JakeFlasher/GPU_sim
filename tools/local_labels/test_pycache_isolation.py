"""Regression coverage for repository-wide adjacent-bytecode isolation."""

from __future__ import annotations

import importlib
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest


def test_adjacent_timestamp_cache_cannot_override_current_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "gpu_simulator_stale_bytecode_probe"
    source = tmp_path / f"{module_name}.py"
    stale_source = 'VALUE = "stale"\n'
    current_source = 'VALUE = "fresh"\n'
    assert len(stale_source) == len(current_source)

    source.write_text(stale_source, encoding="utf-8")
    source_stat = source.stat()
    adjacent_cache = (
        tmp_path
        / "__pycache__"
        / f"{module_name}.{sys.implementation.cache_tag}.pyc"
    )
    adjacent_cache.parent.mkdir()
    py_compile.compile(
        os.fspath(source),
        cfile=os.fspath(adjacent_cache),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    source.write_text(current_source, encoding="utf-8")
    os.utime(
        source,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
    )

    current_stat = source.stat()
    cache_header = adjacent_cache.read_bytes()[:16]
    assert int.from_bytes(cache_header[4:8], "little") == 0
    assert int.from_bytes(cache_header[8:12], "little") == (
        int(current_stat.st_mtime) & 0xFFFF_FFFF
    )
    assert int.from_bytes(cache_header[12:16], "little") == current_stat.st_size
    assert adjacent_cache.is_file()
    assert sys.dont_write_bytecode is True
    assert sys.pycache_prefix is not None
    assert os.environ["PYTHONPYCACHEPREFIX"] == sys.pycache_prefix
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not Path(sys.pycache_prefix).exists()

    monkeypatch.syspath_prepend(os.fspath(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
        assert module.VALUE == "fresh"
        assert module.__cached__ != os.fspath(adjacent_cache)

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {module_name}; print({module_name}.VALUE)",
            ],
            cwd=tmp_path,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == "fresh\n"
        assert completed.stderr == ""
        assert not Path(sys.pycache_prefix).exists()
    finally:
        sys.modules.pop(module_name, None)
