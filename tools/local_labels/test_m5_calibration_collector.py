from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest


COLLECTOR_PATH = Path(__file__).with_name("m5_calibration_collector.py")


def test_direct_refusal_is_the_first_executable_module_action() -> None:
    module = ast.parse(COLLECTOR_PATH.read_text(encoding="utf-8"))
    bootstrap = module.body[2]

    assert isinstance(module.body[0], ast.Expr)
    assert isinstance(module.body[0].value, ast.Constant)
    assert isinstance(module.body[0].value.value, str)  # module docstring
    assert isinstance(module.body[1], ast.ImportFrom)
    assert module.body[1].module == "__future__"
    assert [(alias.name, alias.asname) for alias in module.body[1].names] == [
        ("annotations", None)
    ]
    assert isinstance(bootstrap, ast.If)
    assert ast.dump(bootstrap.test) == ast.dump(
        ast.parse('__name__ == "__main__"', mode="eval").body
    )
    assert [type(statement) for statement in bootstrap.body] == [
        ast.Import,
        ast.Expr,
        ast.Raise,
    ]
    assert ast.dump(bootstrap.body[0]) == ast.dump(
        ast.parse("import sys as _bootstrap_sys").body[0]
    )
    assert ast.dump(bootstrap.body[1]) == ast.dump(
        ast.parse(
            '_bootstrap_sys.stderr.write("M5_COLLECT_DISABLED\\n")'
        ).body[0]
    )
    assert ast.dump(bootstrap.body[2]) == ast.dump(
        ast.parse("raise SystemExit(2)").body[0]
    )
    expected_import_surface = ast.parse(
        '''
M5_COLLECT_DISABLED = "M5_COLLECT_DISABLED"


class M5CollectorDisabled(RuntimeError):
    """M5 collection was requested before its external gates were satisfied."""


def collect(*_args: object, **_kwargs: object) -> None:
    """Refuse collection without reading inputs or creating outputs."""

    raise M5CollectorDisabled(M5_COLLECT_DISABLED)
'''
    ).body
    assert ast.dump(
        ast.Module(body=module.body[3:], type_ignores=[])
    ) == ast.dump(ast.Module(body=expected_import_surface, type_ignores=[]))


def test_collect_refuses_without_creating_output(tmp_path: Path) -> None:
    collector = importlib.import_module(
        "tools.local_labels.m5_calibration_collector"
    )
    output = tmp_path / "never-created"

    with pytest.raises(collector.M5CollectorDisabled) as refusal:
        collector.collect(output=output)

    assert str(refusal.value) == "M5_COLLECT_DISABLED"
    assert not output.exists()


def test_direct_collector_refuses_before_argument_handling(tmp_path: Path) -> None:
    output = tmp_path / "never-created"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(COLLECTOR_PATH.resolve()),
            "--output",
            str(output),
            "--unknown-live-option",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "M5_COLLECT_DISABLED\n"
    assert not output.exists()
