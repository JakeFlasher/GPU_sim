"""Contract tests for the tracked SOL pattern-mapping authority."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAFFOLD_ROOT = REPO_ROOT / "tools" / "sol_baselines_scaffold"
CANONICAL_MAPPING = SCAFFOLD_ROOT / "sol_pattern_mapping.md"
LEGACY_MAPPING_TEXT = "/".join((".humanize", "plans", "sol_pattern_mapping.md"))


def _load_generate_module():
    module_path = SCAFFOLD_ROOT / "generate.py"
    spec = importlib.util.spec_from_file_location(
        "sol_mapping_contract_generate", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_mapping_is_tracked_canonical_contract() -> None:
    generate = _load_generate_module()

    assert generate.MAPPING_PATH == CANONICAL_MAPPING
    assert generate.build_arg_parser().parse_args([]).mapping_path == CANONICAL_MAPPING
    assert CANONICAL_MAPPING.is_file()
    assert len(generate.parse_mapping(CANONICAL_MAPPING)) == 235


def test_live_consumers_have_no_humanize_fallback() -> None:
    consumers = (
        SCAFFOLD_ROOT / "generate.py",
        SCAFFOLD_ROOT / "scaffold.py",
        SCAFFOLD_ROOT / "run_batch.py",
        REPO_ROOT / "tools" / "sol_kb" / "generate_dossiers.py",
    )

    for consumer in consumers:
        source = consumer.read_text(encoding="utf-8")
        assert LEGACY_MAPPING_TEXT not in source, consumer
        assert "sol_baselines_scaffold" in source, consumer
        assert "sol_pattern_mapping.md" in source, consumer
