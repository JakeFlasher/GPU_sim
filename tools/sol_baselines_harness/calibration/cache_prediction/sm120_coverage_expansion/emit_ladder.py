"""Build the sm_120 coverage-readiness ladder manifest from
disk evidence and the closed-vocabulary reducer.

Run directly:

    python -m baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.emit_ladder \\
        --output scripts/baselines_harness/calibration/cache_prediction/sm120_coverage_readiness_ladder_v1.toml \\
        [--captured-at-utc 2026-05-27T08:00:00+00:00]

Determinism: with the same disk state AND the same explicit
``--captured-at-utc`` value, re-running the emitter produces a
byte-identical TOML. The deterministic re-emission test in
``tests/test_sm120_coverage_readiness_ladder.py`` exercises this
contract by reusing the manifest's own stamped timestamp on the
second run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[4]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.disk_discovery import (
    DiscoveryResult,
    build_diagnostics_and_env,
    probe_disk_state,
)
from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_qa import (
    compute_corpus_sha256,
    validate_ladder_manifest,
)
from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_reducer import (
    emit_ladder_state,
)
from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_schema import (
    LADDER_HOST_ARCH,
    LADDER_SCHEMA_VERSION,
    LADDER_TOTAL_PROBLEMS,
    LadderEntry,
    LadderManifest,
    LadderManifestMeta,
)


_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "sm120_coverage_readiness_ladder_v1.toml"
)


def build_manifest(
    *,
    captured_at_utc: str,
    discovery: DiscoveryResult | None = None,
) -> LadderManifest:
    """Build a ``LadderManifest`` from disk evidence.

    The reducer is invoked for every one of the 235 problem slugs.
    Slugs whose harvest pipeline has not touched them pass an
    empty diagnostics dict and fall through to ``spec_only``.
    """
    discovery_resolved: DiscoveryResult = (
        discovery if discovery is not None else probe_disk_state()
    )
    if len(discovery_resolved.corpus_slugs) != LADDER_TOTAL_PROBLEMS:
        raise ValueError(
            f"discovery corpus has {len(discovery_resolved.corpus_slugs)} "
            f"slugs; expected exactly {LADDER_TOTAL_PROBLEMS}"
        )
    diagnostics, env = build_diagnostics_and_env(discovery_resolved)
    entries = tuple(
        LadderEntry(
            slug=slug,
            state=emit_ladder_state(slug, diagnostics[slug], env),
        )
        for slug in discovery_resolved.corpus_slugs
    )
    meta = LadderManifestMeta(
        schema_version=LADDER_SCHEMA_VERSION,
        host_arch=LADDER_HOST_ARCH,
        captured_at_utc=captured_at_utc,
        total_problems=LADDER_TOTAL_PROBLEMS,
        corpus_sha256=compute_corpus_sha256(
            list(discovery_resolved.corpus_slugs)
        ),
    )
    return LadderManifest(meta=meta, entries=entries)


def render_manifest_toml(manifest: LadderManifest) -> str:
    """Render a ``LadderManifest`` to its canonical TOML string.

    Output is deterministic: tab-character-free, two-space indent
    not used (TOML is flat), entries sorted by slug, exactly one
    blank line between sections. The bytes this writer emits are
    what the QA validator and the determinism test consume.
    """
    lines: list[str] = []
    lines.append("# sm_120 coverage-readiness ladder manifest.")
    lines.append("#")
    lines.append(
        "# Tags every one of the 235 SOL-ExecBench problems with "
        "its highest"
    )
    lines.append(
        "# achieved state on the live RTX 5060 Laptop / GB206 host."
    )
    lines.append(
        "# Closed vocabulary: see ``_LADDER_STATES`` in"
    )
    lines.append(
        "# ``scripts/baselines_harness/calibration/cache_prediction/"
    )
    lines.append(
        "# sm120_coverage_expansion/ladder_states.py``."
    )
    lines.append("")
    lines.append("[meta]")
    lines.append(f'schema_version = "{manifest.meta.schema_version}"')
    lines.append(f'host_arch = "{manifest.meta.host_arch}"')
    lines.append(f'captured_at_utc = "{manifest.meta.captured_at_utc}"')
    lines.append(f"total_problems = {manifest.meta.total_problems}")
    lines.append(f'corpus_sha256 = "{manifest.meta.corpus_sha256}"')
    lines.append("")
    for entry in manifest.entries:
        lines.append("[[entries]]")
        lines.append(f'slug = "{entry.slug}"')
        lines.append(f'state = "{entry.state}"')
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def emit_coverage_readiness_ladder(
    *,
    output: Path,
    captured_at_utc: str,
    discovery: DiscoveryResult | None = None,
) -> Path:
    """Build the manifest, render it, write to ``output``, then
    re-read and validate.

    Returns the output path on success. Raises ``LadderQaError``
    on validation failure (which is a self-test: the emitter must
    never write a manifest the QA validator would reject).
    """
    discovery_resolved: DiscoveryResult = (
        discovery if discovery is not None else probe_disk_state()
    )
    manifest = build_manifest(
        captured_at_utc=captured_at_utc,
        discovery=discovery_resolved,
    )
    toml_text = render_manifest_toml(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml_text, encoding="utf-8")
    validate_ladder_manifest(
        output,
        expected_corpus_slugs=list(discovery_resolved.corpus_slugs),
    )
    return output


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit sm120_coverage_readiness_ladder_v1.toml from disk "
            "evidence (235-problem corpus + landed baselines + R11 "
            "calibration artifacts)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Output path. Defaults to the canonical location at "
            "scripts/baselines_harness/calibration/cache_prediction/"
            "sm120_coverage_readiness_ladder_v1.toml"
        ),
    )
    parser.add_argument(
        "--captured-at-utc",
        type=str,
        default=None,
        help=(
            "ISO-8601 UTC timestamp to stamp into meta. Defaults "
            "to the current UTC time at second precision. Re-runs "
            "with the same timestamp produce byte-identical TOML."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    captured_at_utc = args.captured_at_utc or datetime.now(
        timezone.utc
    ).replace(microsecond=0).isoformat()
    output = emit_coverage_readiness_ladder(
        output=args.output,
        captured_at_utc=captured_at_utc,
    )
    print(f"emitted: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
