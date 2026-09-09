"""Resolve the current tool packages for direct entrypoint execution."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def install_package_aliases() -> None:
    """Expose the compatibility package names used by retained modules."""

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    aliases = {
        "baselines_harness": "tools.sol_baselines_harness",
        "sol_baselines_harness": "tools.sol_baselines_harness",
        "baselines_scaffold": "tools.sol_baselines_scaffold",
    }
    for alias, target in aliases.items():
        if alias not in sys.modules:
            sys.modules[alias] = importlib.import_module(target)
