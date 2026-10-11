#!/usr/bin/env python3
"""Strip NCU log chatter and retain the raw CSV header/units/data rows."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lines = args.input.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith('"ID"') or line.startswith("ID,")), None)
    payload = "\n".join(lines[start:]) + "\n" if start is not None else ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    print(f"rows_bytes={len(payload.encode('utf-8'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
