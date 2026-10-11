#!/usr/bin/env python3
"""Recover nvdisasm text streams from DocumentSASS's binary memcpy capture.

CUDA 13.2 exposes OPERATION SETS in multiple memcpy fragments but need not
expose the historical ARCHITECTURE header. Do not pipe the interception through
strings: doing so discards bytes and invalidates each record's length.
The output is tool scheduling metadata, not a measured hardware latency table.
"""
import argparse
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--marker", default="OPERATION SETS")
    args = ap.parse_args()
    data = args.capture.read_bytes()
    header = re.compile(rb"\n<(0x[0-9a-f]+) (0x[0-9a-f]+) (\d+)>\n")
    records = []
    cursor = 0
    while match := header.search(data, cursor):
        end = match.end() + int(match[3])
        if end > len(data):
            raise RuntimeError("truncated memcpy payload; retain raw binary capture")
        records.append((match[2], data[match.end():end]))
        cursor = end
    marker = args.marker.encode("ascii")
    source = next((src for src, payload in records
                   if payload.lstrip().startswith(marker)), None)
    if source is None:
        raise RuntimeError(f"no captured stream starts with {args.marker!r}")
    # DocumentSASS's source-buffer reconstruction: serialized copies from the
    # same parser input buffer supply successive fragments in capture order.
    parts = [payload for src, payload in records if src == source]
    text = b"".join(parts).decode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"Recovered {len(text)} bytes in {len(parts)} fragments to {args.output}")


if __name__ == "__main__":
    main()
