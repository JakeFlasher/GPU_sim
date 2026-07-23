#!/usr/bin/env python3
"""Transcribe measured pchase latencies into an ArchProfile card (plan §1D).

Reads a ``pchase_static_card`` artifact (the counter-free clock64-measured
evidence) and writes/refreshes the ``latencies_cycles`` block of the matching
``arch_profiles/cards/<arch>.json``. The ArchProfile is the canonical per-arch
description ``sim-calib`` consumes; the pchase card remains the measurement
evidence, and ``test_arch_profile_lockstep.py`` asserts the two never drift.

Rule-1 note: these are microbench-DERIVED constants (clock64 timing inside
probe kernels), not NCU counter values — they are legitimate simulator inputs.

Fail-closed contract:
- every latency property must be present with kind == "value";
- probes the pchase harness could not resolve (kind == "typed") map to the
  ArchProfile ``research_gap`` object, never to a guessed number;
- arch mismatch between the two cards aborts.

Usage:
  .venv/bin/python tools/pchase/transcribe_latencies_to_arch_card.py \
      --pchase-card data/pchase/sm_86_static_card.json \
      --arch-card arch_profiles/cards/sm_86.json \
      --host-clock-note "host-locked 1350 MHz (nvidia-smi -lgc), persistence on"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pchase-card", required=True)
    ap.add_argument("--arch-card", required=True)
    ap.add_argument("--host-clock-note", default="")
    args = ap.parse_args()

    pchase_path = Path(args.pchase_card)
    arch_path = Path(args.arch_card)
    pchase = json.loads(pchase_path.read_text())
    arch = json.loads(arch_path.read_text())

    if pchase.get("artifact_kind") != "pchase_static_card":
        print(f"not a pchase_static_card: {pchase_path}", file=sys.stderr)
        return 2
    if pchase.get("arch") != arch.get("arch_id"):
        print(
            f"arch mismatch: pchase card {pchase.get('arch')!r} vs "
            f"ArchProfile {arch.get('arch_id')!r}",
            file=sys.stderr,
        )
        return 2

    props = pchase.get("properties", {})
    block: dict = {}
    for card_key, prop_key in LATENCY_MAP.items():
        rec = props.get(prop_key)
        if not rec or rec.get("kind") != "value":
            print(
                f"latency property {prop_key} missing or unresolved "
                f"(kind={rec.get('kind') if rec else None}) — refusing to "
                f"transcribe a partial latency block",
                file=sys.stderr,
            )
            return 3
        block[card_key] = {
            "value": rec["value"],
            "dispersion": rec.get("dispersion"),
            "unit": rec.get("unit", "cycles_per_dependent_load"),
        }
    for card_key, prop_key in GAP_MAP.items():
        rec = props.get(prop_key)
        if rec is None:
            continue
        if rec.get("kind") == "value":
            block[card_key] = {
                "value": rec["value"],
                "dispersion": rec.get("dispersion"),
                "unit": rec.get("unit"),
            }
        else:
            block[card_key] = {
                "status": "research_gap",
                "reason": rec.get("detail", rec.get("typed_reason", "unresolved")),
                "challenge": "pchase probe limitation, not DVFS noise — see "
                             "evidence curve in the pchase static card",
            }

    prov = (
        f"{pchase_path.as_posix()} (device {pchase.get('device')}, driver "
        f"{pchase.get('driver_version')}"
    )
    if args.host_clock_note:
        prov += f", {args.host_clock_note}"
    prov += ")"
    block["provenance"] = prov

    arch["latencies_cycles"] = block
    arch_path.write_text(json.dumps(arch, indent=2) + "\n")
    print(f"transcribed {len(LATENCY_MAP)} latencies -> {arch_path}")
    for k in LATENCY_MAP:
        print(f"  {k}: {block[k]['value']} +/- {block[k]['dispersion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
