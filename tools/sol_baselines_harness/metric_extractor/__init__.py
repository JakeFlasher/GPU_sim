"""Strict consumers for ``nvbit_tracer/metric_extractor/`` artifacts.

This package wraps the per-launch JSON-L file the metric extractor
NVBit tool emits to ``METRIC_EXTRACTOR_LAUNCHES_PATH`` (default
``./metric_extractor_launches.jsonl``). The schema lives at
``nvbit_tracer/metric_extractor/JSONL_SCHEMA.md``; the parser here is
the canonical strict consumer that the harness surface plugin in a
later round will use to ingest records.
"""

from .launches_parser import (  # noqa: F401
    LaunchRecord,
    ParseError,
    parse_launches_jsonl,
)
from .calibration_harness import (  # noqa: F401
    CalibrationError,
    CalibrationReport,
    KernelResult,
    MetricVerdict,
    run_calibration,
)
