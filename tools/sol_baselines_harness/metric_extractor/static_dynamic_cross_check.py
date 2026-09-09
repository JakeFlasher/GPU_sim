"""Durable static/dynamic SASS cross-check for the Blackwell buckets.

Compares the static SASS scan's TCGen05 / TMA / DSMEM mnemonic counts
against the dynamic ``instruction_class_histogram`` totals from a
representative ``launches.jsonl`` bundle. The earlier local audit
was structural — it asserted ``tcgen05 == tma == dsmem == 0``
without recording the static scan inputs or the architecture
constraint. This module owns the durable artifact:

* Takes a static SASS source path (typically a ``ptxas -v`` dump or
  a ``cuobjdump --dump-sass`` output) and a launches bundle path.
* Scans the SASS text for TCGen05 / TMA / DSMEM mnemonics; counts the
  occurrences.
* Sums the dynamic bucket counts from every record in the bundle.
* Records the architecture constraint resolved from the bundle's
  ``host.compute_capability``. On a pre-Blackwell device,
  Blackwell-only mnemonics cannot appear in the static scan by
  hardware constraint, so a zero-vs-zero comparison is the correct
  verdict — but the artifact still has to record the constraint
  explicitly so an operator scanning the report sees WHY the zeros
  are acceptable.
* Emits a structured JSON dict with the static counts, dynamic
  counts, a per-bucket pass/fail verdict, the resolved architecture,
  the input file paths + SHA-256 hashes, and an overall verdict.

The artifact is consumed by review evidence and by future B200
cross-check work (the same comparison runs on B200 against the same
bundle shape; the artifact format stays stable across hosts).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Canonical bucket -> mnemonic-prefix regex map. Each prefix matches the
# canonical SASS tokens NVBit's ``classify_mnemonic()`` routes into the
# bucket (see ``nvbit_tracer/metric_extractor/classifier.h`` for the
# authoritative routing). Keeping the regex deliberately conservative —
# we only count tokens that are unambiguous representatives of each
# bucket, so an operator reading the audit can tell exactly which
# mnemonics contributed.
_BUCKET_MNEMONIC_PATTERNS = {
    "tcgen05": [
        re.compile(r"\bTCGEN05\.[A-Z0-9_:.]+"),
    ],
    "tma": [
        # UTMACCTL / UTMACCTLR / UTMALDG / UTMASTG variants are the
        # canonical Blackwell SASS encodings for cp.async.bulk.tensor.
        re.compile(r"\bUTMA[A-Z]*[A-Z0-9_]*"),
    ],
    "dsmem": [
        # LDS.MMIO / STS.MMIO are the canonical SASS for distributed
        # shared memory access.
        re.compile(r"\b(?:LDS|STS)\.MMIO\b"),
    ],
}

_ARCH_BLACKWELL_ONLY_BUCKETS = ("tcgen05", "tma", "dsmem")


@dataclass
class StaticScanResult:
    """One scan over a single SASS file."""

    source_path: str
    source_sha256: str
    source_bytes: int
    bucket_counts: Dict[str, int] = field(default_factory=dict)
    # Up to 16 representative matches per bucket; truncated so the
    # JSON artifact stays compact while still proving "these are the
    # exact mnemonics that contributed".
    bucket_samples: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DynamicAggregateResult:
    """Aggregate of every record in a launches.jsonl bundle."""

    bundle_path: str
    bundle_sha256: str
    records_seen: int
    bucket_sums: Dict[str, int] = field(default_factory=dict)


@dataclass
class CrossCheckVerdict:
    """Per-bucket pass/fail verdict + architecture context."""

    bucket: str
    static_count: int
    dynamic_count: int
    verdict: str  # "pass" | "fail" | "skip"
    reason: Optional[str] = None


@dataclass
class CrossCheckReport:
    """End-to-end T33 cross-check artifact."""

    static: StaticScanResult
    dynamic: DynamicAggregateResult
    arch_label: Optional[str]
    compute_capability: Optional[str]
    architecture_constraint: Optional[str]
    verdicts: List[CrossCheckVerdict] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return all(v.verdict in ("pass", "skip") for v in self.verdicts)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def scan_static_sass(sass_path: Path) -> StaticScanResult:
    """Scan a SASS text file for TCGen05 / TMA / DSMEM mnemonics. Each
    bucket carries the count and up to 16 sample matches. A missing
    or unreadable file raises ``FileNotFoundError``; an empty file
    contributes zero to every bucket.
    """
    raw = sass_path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    result = StaticScanResult(
        source_path=str(sass_path),
        source_sha256=_sha256_bytes(raw),
        source_bytes=len(raw),
    )
    for bucket, patterns in _BUCKET_MNEMONIC_PATTERNS.items():
        matches: List[str] = []
        for pat in patterns:
            matches.extend(pat.findall(text))
        result.bucket_counts[bucket] = len(matches)
        # Sample up to 16 distinct mnemonics so the artifact stays
        # compact and the audit can show which mnemonics contributed
        # without dumping the entire match set.
        unique_matches: List[str] = []
        seen = set()
        for m in matches:
            if m in seen:
                continue
            seen.add(m)
            unique_matches.append(m)
            if len(unique_matches) >= 16:
                break
        result.bucket_samples[bucket] = unique_matches
    return result


def aggregate_dynamic_buckets(
    launches_path: Path,
) -> DynamicAggregateResult:
    """Aggregate the dynamic ``instruction_class_histogram`` counts
    across every record in a ``launches.jsonl`` bundle for the
    Blackwell-specific instruction buckets.
    """
    try:
        raw = launches_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"{launches_path}: cannot read launches.jsonl: {exc}"
        ) from exc
    result = DynamicAggregateResult(
        bundle_path=str(launches_path),
        bundle_sha256=_sha256_bytes(raw),
        records_seen=0,
    )
    # Initialise the bucket totals so a bundle with zero records
    # produces explicit zeros rather than a missing key.
    for bucket in _ARCH_BLACKWELL_ONLY_BUCKETS:
        result.bucket_sums[bucket] = 0
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{launches_path}: launches.jsonl is not valid UTF-8"
        ) from exc
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{launches_path}:{line_number}: malformed JSONL record: "
                f"{exc.msg}"
            ) from exc
        if not isinstance(rec, dict):
            raise ValueError(
                f"{launches_path}:{line_number}: JSONL record must be an object"
            )
        result.records_seen += 1
        hist = rec.get("instruction_class_histogram")
        if not isinstance(hist, dict):
            continue
        for bucket in _ARCH_BLACKWELL_ONLY_BUCKETS:
            val = hist.get(bucket, 0)
            if isinstance(val, int) and not isinstance(val, bool):
                result.bucket_sums[bucket] += val
    return result


def _resolve_compute_capability(launches_path: Path) -> tuple:
    """Read the first record's ``host`` block to extract the
    ``compute_capability`` + ``arch_label``. The bundle's host
    fingerprint is uniform across every record so a single read is
    authoritative.
    """
    with launches_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            host = rec.get("host")
            if isinstance(host, dict):
                return (
                    host.get("compute_capability"),
                    host.get("arch_label"),
                )
            return None, None
    return None, None


def _architecture_constraint(compute_capability: Optional[str]) -> Optional[str]:
    """Resolve the architecture constraint that explains zero-vs-zero
    matches on a non-Blackwell host. Returns ``None`` for Blackwell
    targets where the buckets are expected to be non-zero.
    """
    if not compute_capability:
        return None
    # Strip the version suffix; sm_86 -> "8.6" or "8.6a" both
    # collapse to the major.minor pair.
    cc_clean = compute_capability.lower().strip()
    # Blackwell datacenter = compute_capability "10.0" / "10.0a".
    # Blackwell consumer = "12.0". Everything else is pre-Blackwell.
    if cc_clean.startswith("10.") or cc_clean.startswith("12."):
        return None
    return (
        f"compute_capability={compute_capability!r} is pre-Blackwell; "
        "the TCGen05 / TMA / DSMEM mnemonics this audit checks for are "
        "Blackwell-only SASS encodings (sm_100a / sm_120). The compiler "
        "cannot emit them on this host; both the static scan and the "
        "dynamic aggregation are expected to return zero."
    )


def cross_check_static_vs_dynamic(
    sass_path: Path,
    launches_path: Path,
) -> CrossCheckReport:
    """Build the full T33 cross-check artifact.

    Returns a :class:`CrossCheckReport` with per-bucket verdicts and
    architecture context. Verdict semantics:

    * ``pass`` — static count and dynamic count agree (both zero or
      both positive). On Blackwell hosts the bucket should be
      positive; on pre-Blackwell hosts zero is the correct
      observation and the architecture constraint records why.
    * ``fail`` — static count is non-zero but dynamic count is zero,
      or vice versa. This is the misclassification signal the audit
      exists to surface.
    * ``skip`` — pre-Blackwell host (compute_capability < 10.0)
      where both counts are zero by hardware constraint; the verdict
      records that the audit ran but the comparison is structurally
      uninformative on this host.
    """
    static_result = scan_static_sass(sass_path)
    dynamic_result = aggregate_dynamic_buckets(launches_path)
    cc, arch_label = _resolve_compute_capability(launches_path)
    constraint = _architecture_constraint(cc)
    report = CrossCheckReport(
        static=static_result,
        dynamic=dynamic_result,
        arch_label=arch_label,
        compute_capability=cc,
        architecture_constraint=constraint,
    )
    for bucket in _ARCH_BLACKWELL_ONLY_BUCKETS:
        s = static_result.bucket_counts.get(bucket, 0)
        d = dynamic_result.bucket_sums.get(bucket, 0)
        if constraint is not None and s == 0 and d == 0:
            # Pre-Blackwell host, both zero — skip with constraint
            # reason so the audit log shows the comparison was
            # acknowledged rather than silently treated as a pass.
            verdict = "skip"
            reason = constraint
        elif s == 0 and d == 0:
            # Blackwell host, both zero — that's unexpected (the
            # corpus should produce at least one TCGen05/TMA/DSMEM
            # entry). The negative-test contract maps this to
            # "unexpected_zero_count for human review".
            verdict = "fail"
            reason = (
                f"both static and dynamic counts are zero on a "
                f"Blackwell-targeted host (compute_capability={cc!r}); "
                "this is the unexpected_zero_count signal that the "
                "audit exists to surface."
            )
        elif s > 0 and d == 0:
            verdict = "fail"
            reason = (
                f"static scan found {s} {bucket} mnemonic(s) but "
                "dynamic aggregation observed zero — the classifier "
                "may be routing these mnemonics into a different bucket."
            )
        elif s == 0 and d > 0:
            verdict = "fail"
            reason = (
                f"static scan found zero {bucket} mnemonics but "
                f"dynamic aggregation observed {d} — the classifier "
                "may be routing some other mnemonic into this bucket."
            )
        else:
            verdict = "pass"
            reason = None
        report.verdicts.append(CrossCheckVerdict(
            bucket=bucket,
            static_count=s,
            dynamic_count=d,
            verdict=verdict,
            reason=reason,
        ))
    return report


def report_to_dict(report: CrossCheckReport) -> Dict[str, Any]:
    """Serialise the report into a plain-dict JSON shape. Useful for
    writing the artifact and for round-tripping in tests.
    """
    return {
        "static": asdict(report.static),
        "dynamic": asdict(report.dynamic),
        "arch_label": report.arch_label,
        "compute_capability": report.compute_capability,
        "architecture_constraint": report.architecture_constraint,
        "all_pass": report.all_pass,
        "verdicts": [asdict(v) for v in report.verdicts],
    }
