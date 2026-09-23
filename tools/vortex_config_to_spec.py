#!/usr/bin/env python3
"""Import literal Vortex execution shape; never infer timing from bank counts.

An explicitly resolved issue width is required if the input uses an expression.
This is a structural importer, not a Vortex functional/timing backend.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import tomllib


def literal(table: dict, key: str, maximum: int = 2**32 - 1) -> int:
    value = table.get(key)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{key}: expected a positive resolved integer, got {value!r}")
    return value


def convert(raw: dict, issue_width: int | None = None) -> dict:
    pipeline = raw.get("pipeline") if isinstance(raw, dict) else None
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline: expected a TOML table")
    resolved = literal(pipeline, 'VX_CFG_ISSUE_WIDTH') if issue_width is None else literal({'width': issue_width}, 'width')
    return {
        'schema_version': 1, 'architecture': 'vortex_config', 'isa': 'riscv_vortex',
        'instruction_set': [],
        'execution': {
            'simt_width': literal(pipeline, 'VX_CFG_NUM_THREADS', 64),
            'max_resident_groups': literal(pipeline, 'VX_CFG_NUM_WARPS'),
        },
        # No mapping exists from platform memory banks to lane requests/cycle.
        'resources': [{'name': 'dispatch', 'kind': 'custom', 'issue_width': resolved, 'latency_cycles': None}],
        'evidence': 'unvalidated',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--resolved-issue-width', type=int, help='value from Vortex gen_config; no expression guessing')
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    try:
        raw = tomllib.loads(args.config.read_text(encoding='utf-8'))
        spec = convert(raw, args.resolved_issue_width)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    encoded = json.dumps(spec, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
