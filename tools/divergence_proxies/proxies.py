"""Divergence + MSHR-pressure proxy module (C-AC-G).

Consumes a per-kernel-per-launch evidence bundle and emits the
8 named proxies the original-plan AC9 enumeration pins:

1. ``unique_sectors_per_warp_inst``
2. ``unique_lines_per_warp_inst``
3. ``sectors_per_request``
4. ``md_loads_per_kilo_inst``
5. ``outstanding_load_density_by_phase``
6. ``pc_repeated_miss_likely_streams``
7. ``stride_entropy``
8. ``throughput_saturation_slope_under_resident_warp_sweep``

Each proxy is either:

* an **admitted numeric record** ``{"result_kind":
  "proxy_value", "value": <float>}`` with optional
  ``sample_count``; OR
* a **typed-unavailable record** ``{"result_kind":
  "typed_unavailable", "typed_reason": <enum>}`` when the
  trace did not carry the evidence the proxy requires.

The typed-unavailable discipline is the C-AC-G negative-test
contract: never emit silent zero defaults when the underlying
evidence is absent, because zero is a legitimate measurement
value for several proxies (e.g., ``md_loads_per_kilo_inst == 0``
on a kernel with no memory-divergent loads is meaningfully
distinct from "no instruction count available").

The output bundle admits via
``workloads/simbench_bw/schemas/divergence_proxies.schema.json``
under strict ``jsonschema.Draft202012Validator`` validation.
The strict-schema lesson from R7 applies: the producer must
emit the bytes the schema declares, not a sanitized copy.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


# Typed-unavailable taxonomy. One of these
# (`outstanding_load_density_unavailable_no_phase_records`)
# is pinned in the original-plan AC19 enumeration; the other
# seven are round-added per the C-AC-C lockstep discipline.
# The frozenset is the producer-side surface in the
# divergence-proxies taxonomy family declared in
# `tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`.
DIVERGENCE_PROXIES_TYPED_REASONS: frozenset[str] = frozenset({
    "outstanding_load_density_unavailable_no_phase_records",
    "stride_entropy_unavailable_no_lane_table",
    "throughput_saturation_slope_unavailable_no_sweep",
    "pc_repeated_miss_likely_streams_unavailable_no_pc_data",
    "md_loads_per_kilo_inst_unavailable_no_instruction_count",
    "unique_sectors_per_warp_inst_unavailable_no_records",
    "unique_lines_per_warp_inst_unavailable_no_records",
    "sectors_per_request_unavailable_no_records",
})


# The 8 proxy names the bundle must always carry. The order
# here is the canonical AC9 enumeration order; the bundle's
# JSON keys appear in this exact order.
PROXY_NAMES: tuple[str, ...] = (
    "unique_sectors_per_warp_inst",
    "unique_lines_per_warp_inst",
    "sectors_per_request",
    "md_loads_per_kilo_inst",
    "outstanding_load_density_by_phase",
    "pc_repeated_miss_likely_streams",
    "stride_entropy",
    "throughput_saturation_slope_under_resident_warp_sweep",
)


# The canonical SASS stride bucket set used by ``stride_entropy``.
# Strides outside the set fold into the ``other`` bucket. The
# bucketization is deliberate: the SASS LD/ST byte-size set is
# {1, 2, 4, 8, 16}, common 32B / 128B sector / line strides
# correspond to coalesced patterns, and the 0-stride bucket
# captures the broadcast case.
_STRIDE_BUCKETS: tuple[int, ...] = (0, 4, 8, 16, 32, 64, 128)


def _admitted(value: float, sample_count: int | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "result_kind": "proxy_value",
        "value": float(value),
    }
    if sample_count is not None:
        rec["sample_count"] = int(sample_count)
    return rec


def _unavailable(typed_reason: str) -> dict[str, Any]:
    if typed_reason not in DIVERGENCE_PROXIES_TYPED_REASONS:
        raise ValueError(
            f"Unknown typed_reason {typed_reason!r}; not in "
            f"DIVERGENCE_PROXIES_TYPED_REASONS"
        )
    return {
        "result_kind": "typed_unavailable",
        "typed_reason": typed_reason,
    }


def _is_admitted_sector_stream_record(rec: dict[str, Any]) -> bool:
    return rec.get("result_kind") == "sector_stream"


def _proxy_unique_sectors_per_warp_inst(
    sector_stream_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    admitted = [
        r for r in sector_stream_records
        if _is_admitted_sector_stream_record(r)
    ]
    if not admitted:
        return _unavailable(
            "unique_sectors_per_warp_inst_unavailable_no_records"
        )
    total = sum(int(r["touched_32B_sectors"]) for r in admitted)
    return _admitted(
        total / len(admitted), sample_count=len(admitted)
    )


def _proxy_unique_lines_per_warp_inst(
    sector_stream_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    admitted = [
        r for r in sector_stream_records
        if _is_admitted_sector_stream_record(r)
    ]
    if not admitted:
        return _unavailable(
            "unique_lines_per_warp_inst_unavailable_no_records"
        )
    total = sum(int(r["touched_128B_lines"]) for r in admitted)
    return _admitted(
        total / len(admitted), sample_count=len(admitted)
    )


def _proxy_sectors_per_request(
    sector_stream_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    admitted = [
        r for r in sector_stream_records
        if _is_admitted_sector_stream_record(r)
    ]
    if not admitted:
        return _unavailable(
            "sectors_per_request_unavailable_no_records"
        )
    total = sum(float(r["sectors_per_request"]) for r in admitted)
    return _admitted(
        total / len(admitted), sample_count=len(admitted)
    )


def _proxy_md_loads_per_kilo_inst(
    sector_stream_records: Sequence[dict[str, Any]],
    total_instruction_count: int | None,
) -> dict[str, Any]:
    if total_instruction_count is None or total_instruction_count <= 0:
        return _unavailable(
            "md_loads_per_kilo_inst_unavailable_no_instruction_count"
        )
    admitted = [
        r for r in sector_stream_records
        if _is_admitted_sector_stream_record(r)
    ]
    md_count = sum(
        1 for r in admitted
        if int(r["touched_32B_sectors"]) > 1
    )
    return _admitted(
        1000.0 * md_count / float(total_instruction_count),
        sample_count=total_instruction_count,
    )


def _proxy_outstanding_load_density_by_phase(
    phase_records: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not phase_records:
        return _unavailable(
            "outstanding_load_density_unavailable_no_phase_records"
        )
    # Approximation: per-phase load-issue rate, averaged across
    # phases. Each phase carries `(start_clock, end_clock,
    # load_count)`; the rate is `load_count / max(1, span)`. The
    # final proxy is the arithmetic mean of per-phase rates. NOT
    # true MSHR occupancy.
    rates: list[float] = []
    for p in phase_records:
        span = max(
            1, int(p["end_clock"]) - int(p["start_clock"])
        )
        rates.append(float(p["load_count"]) / float(span))
    if not rates:
        return _unavailable(
            "outstanding_load_density_unavailable_no_phase_records"
        )
    return _admitted(
        sum(rates) / len(rates), sample_count=len(rates)
    )


def _proxy_pc_repeated_miss_likely_streams(
    sector_stream_records: Sequence[dict[str, Any]],
    pc_per_record: Sequence[int | None] | None,
) -> dict[str, Any]:
    """Count distinct PCs whose every appearance had
    touched_32B_sectors > 1. A "stream-like miss" PC is one
    where every observation across the kernel launch shows the
    same divergent / scattered pattern, which is consistent
    with a stream that always misses cache. The proxy is the
    count of such PCs.
    """
    if pc_per_record is None or len(pc_per_record) != len(sector_stream_records):
        return _unavailable(
            "pc_repeated_miss_likely_streams_unavailable_no_pc_data"
        )
    pc_to_observations: dict[int, list[int]] = {}
    for rec, pc in zip(sector_stream_records, pc_per_record):
        if pc is None or not _is_admitted_sector_stream_record(rec):
            continue
        pc_to_observations.setdefault(pc, []).append(
            int(rec["touched_32B_sectors"])
        )
    if not pc_to_observations:
        return _unavailable(
            "pc_repeated_miss_likely_streams_unavailable_no_pc_data"
        )
    stream_like_pc_count = sum(
        1 for sectors in pc_to_observations.values()
        if len(sectors) >= 2 and all(s > 1 for s in sectors)
    )
    return _admitted(
        float(stream_like_pc_count),
        sample_count=len(pc_to_observations),
    )


def _proxy_stride_entropy(
    lane_vectors: Sequence[Sequence[int]] | None,
) -> dict[str, Any]:
    """Shannon entropy (in bits) of the inter-lane stride
    distribution. Each lane vector contributes
    ``len(vector) - 1`` consecutive-lane strides; strides are
    bucketed into ``_STRIDE_BUCKETS`` (with anything else folded
    into ``other``).
    """
    if lane_vectors is None:
        return _unavailable(
            "stride_entropy_unavailable_no_lane_table"
        )
    bucket_counts: dict[str, int] = {}
    for vector in lane_vectors:
        for i in range(len(vector) - 1):
            stride = vector[i + 1] - vector[i]
            key = (
                str(stride) if stride in _STRIDE_BUCKETS else "other"
            )
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
    total = sum(bucket_counts.values())
    if total == 0:
        return _unavailable(
            "stride_entropy_unavailable_no_lane_table"
        )
    entropy = 0.0
    for count in bucket_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return _admitted(entropy, sample_count=total)


def _proxy_throughput_saturation_slope(
    resident_warp_sweep: dict[str, list[float]] | None,
) -> dict[str, Any]:
    """Least-squares slope of throughput vs. resident-warp-count.
    Requires a sweep of at least 2 launch points; a single-launch
    trace always emits typed-unavailable.
    """
    if resident_warp_sweep is None:
        return _unavailable(
            "throughput_saturation_slope_unavailable_no_sweep"
        )
    xs = resident_warp_sweep.get("resident_warps", [])
    ys = resident_warp_sweep.get("throughput", [])
    if len(xs) != len(ys) or len(xs) < 2:
        return _unavailable(
            "throughput_saturation_slope_unavailable_no_sweep"
        )
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_xx = sum(x * x for x in xs)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return _unavailable(
            "throughput_saturation_slope_unavailable_no_sweep"
        )
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return _admitted(slope, sample_count=n)


def compute_divergence_proxies(
    *,
    manifest_pairing_index: dict[str, str],
    kernel_unique_id: int,
    device: str,
    sector_stream_records: Sequence[dict[str, Any]],
    total_instruction_count: int | None = None,
    phase_records: Sequence[dict[str, Any]] | None = None,
    pc_per_record: Sequence[int | None] | None = None,
    lane_vectors: Sequence[Sequence[int]] | None = None,
    resident_warp_sweep: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Compute the 8-proxy bundle for one kernel launch.

    Inputs
    ------
    ``manifest_pairing_index``
        Provenance index dict carrying the canonical 5-field
        shape (``slug``, ``shape_point``, ``binary_sha256``,
        ``source_commit``, ``kernel_signature_sha256``). Pinned
        in the bundle so downstream verifiers can bind cached
        run-summary fields to on-disk bundle provenance — the
        same shape HRD / sharing locality / order projections /
        perturbation ledger emit. Required.
    ``kernel_unique_id``
        The tracer-assigned unique id for the kernel function
        whose launch produced this evidence. Pinned in the
        bundle for traceability back to the wire record.
    ``device``
        Compute capability tag (``sm_86``, ``sm_120-laptop``).
    ``sector_stream_records``
        Per-warp records emitted by
        :mod:`tools.sector_stream.generator`. Both admitted and
        typed-exclusion records are accepted; only the admitted
        records contribute to the proxy aggregations. The order
        of records does not matter for any proxy.
    ``total_instruction_count``
        Total instruction count across all warps in the
        launch. Required for ``md_loads_per_kilo_inst``;
        absent / non-positive triggers typed-unavailable.
    ``phase_records``
        Optional per-phase records ``{"start_clock", "end_clock",
        "load_count"}``. Required for
        ``outstanding_load_density_by_phase``; absent triggers
        typed-unavailable.
    ``pc_per_record``
        Optional sequence aligned with ``sector_stream_records``
        carrying the PC (``vpc``) per record. Required for
        ``pc_repeated_miss_likely_streams``; absent or
        misaligned-length triggers typed-unavailable.
    ``lane_vectors``
        Optional sequence of 32-element lane-address vectors
        (one per warp memory instruction). Required for
        ``stride_entropy``; absent triggers typed-unavailable.
    ``resident_warp_sweep``
        Optional dict with parallel ``resident_warps`` and
        ``throughput`` lists from a multi-launch sweep. Required
        for ``throughput_saturation_slope_under_resident_warp_sweep``;
        absent or under-2-points triggers typed-unavailable.

    Returns
    -------
    A dict admitting the schema at
    ``workloads/simbench_bw/schemas/divergence_proxies.schema.json``.
    """
    proxies: dict[str, Any] = {
        "unique_sectors_per_warp_inst":
            _proxy_unique_sectors_per_warp_inst(sector_stream_records),
        "unique_lines_per_warp_inst":
            _proxy_unique_lines_per_warp_inst(sector_stream_records),
        "sectors_per_request":
            _proxy_sectors_per_request(sector_stream_records),
        "md_loads_per_kilo_inst":
            _proxy_md_loads_per_kilo_inst(
                sector_stream_records, total_instruction_count
            ),
        "outstanding_load_density_by_phase":
            _proxy_outstanding_load_density_by_phase(phase_records),
        "pc_repeated_miss_likely_streams":
            _proxy_pc_repeated_miss_likely_streams(
                sector_stream_records, pc_per_record
            ),
        "stride_entropy": _proxy_stride_entropy(lane_vectors),
        "throughput_saturation_slope_under_resident_warp_sweep":
            _proxy_throughput_saturation_slope(resident_warp_sweep),
    }

    return {
        "schema_version": 2,
        "manifest_pairing_index": dict(manifest_pairing_index),
        "kernel_unique_id": int(kernel_unique_id),
        "device": str(device),
        **proxies,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI surface: read one JSON evidence record from stdin
    (or ``--input PATH``) and print the proxy bundle on stdout.

    Input shape::

        {
          "manifest_pairing_index": {
              "slug": "<slug>",
              "shape_point": "<shape>",
              "binary_sha256": "<hex>",
              "source_commit": "<hex>",
              "kernel_signature_sha256": "<hex>"
          },
          "kernel_unique_id": <int>,
          "device": "<sm_*>",
          "sector_stream_records": [...],
          "total_instruction_count": <int|null>,
          "phase_records": [...] | null,
          "pc_per_record": [...] | null,
          "lane_vectors": [[...], ...] | null,
          "resident_warp_sweep": {"resident_warps": [...],
                                  "throughput": [...]} | null
        }

    For ad-hoc operator inspection without a real run-summary
    context, an absent ``manifest_pairing_index`` falls back to
    a zero-fingerprint placeholder (the same pattern HRD's CLI
    helper uses); the canonical run_suite.py path always passes
    a real value.
    """
    import argparse
    import json as _json
    import sys

    parser = argparse.ArgumentParser(
        prog="python tools/divergence_proxies/proxies.py",
        description=(
            "Compute the 8-proxy bundle for one kernel launch "
            "(read evidence record from stdin or --input PATH; "
            "write JSON bundle to stdout)."
        ),
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        help="Optional path to a JSON evidence record.",
    )
    args = parser.parse_args(argv)

    if args.input is not None:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = _json.load(f)
    else:
        payload = _json.load(sys.stdin)

    manifest_pairing_index = payload.get("manifest_pairing_index") or {
        "slug": "<unspecified>",
        "shape_point": "<unspecified>",
        "binary_sha256": "0" * 64,
        "source_commit": "0" * 64,
        "kernel_signature_sha256": "0" * 64,
    }
    bundle = compute_divergence_proxies(
        manifest_pairing_index=manifest_pairing_index,
        kernel_unique_id=payload["kernel_unique_id"],
        device=payload["device"],
        sector_stream_records=payload.get(
            "sector_stream_records", []
        ),
        total_instruction_count=payload.get(
            "total_instruction_count"
        ),
        phase_records=payload.get("phase_records"),
        pc_per_record=payload.get("pc_per_record"),
        lane_vectors=payload.get("lane_vectors"),
        resident_warp_sweep=payload.get("resident_warp_sweep"),
    )
    sys.stdout.write(_json.dumps(bundle, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
