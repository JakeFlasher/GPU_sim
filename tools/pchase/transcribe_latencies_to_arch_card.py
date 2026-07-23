#!/usr/bin/env python3
"""Historical p-chase v1 transcriber (retired from production use).

The v1 transcriber is retained for archaeology and refuses before reading a
pchase card, reading an architecture card, or writing any successor output. A
future schema-v2 path must provide verified protocol metadata and an external
immutable publication root before an architecture input can be admitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.producer_retirement import report_legacy_producer_retirement

# ArchProfile key -> pchase properties key
LATENCY_MAP = {
    "l1_hit": "l1_hit_latency_cycles",
    "l2_hit": "l2_hit_latency_cycles",
    "dram": "dram_latency_cycles",
    "shared": "shared_latency_cycles",
}
# Probes transcribed as research_gap when typed-unresolved.
GAP_MAP = {
    "mshr_outstanding_misses": "mshr_outstanding_misses",
}


def main() -> int:
    return report_legacy_producer_retirement(
        "tools/pchase/transcribe_latencies_to_arch_card.py",
        successor_requirement="verified schema-v2 protocol metadata",
    )


if __name__ == "__main__":
    raise SystemExit(main())
