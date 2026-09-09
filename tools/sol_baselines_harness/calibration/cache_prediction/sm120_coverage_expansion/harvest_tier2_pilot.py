"""Per-pilot Tier-2 NVBit harvest driver with batching, resume,
and per-cell failure logging.

The lower-level Tier-2 harvest scaffold lives at
``sm120_l2_associativity_probe/harvest_tier2_traces.py``; that
script targets the calibration smoke binaries (built into
``nvbit_tracer/metric_extractor/tests/calibration/sm120/``).
This driver is the pilot-cohort wrapper: it iterates over the
12 selected pilot cells (or a caller-provided subset), checks
for existing per-cell sidecars (resume semantics), invokes a
pluggable ``harvest_fn`` to actually run the harvest, captures
per-cell failure reasons from a closed vocabulary, parses the
resulting sidecar to extract the per-cell evidence fields
(SHA-256, record count, launch count, byte budget used,
truncation-relaxation flag), and emits a status TOML that the
upstream diagnostics emitter consumes when it refreshes the
per-cell ``tier2_*`` fields.

Two ``harvest_fn`` choices ship in-module:

* ``_stub_harvest_fn`` — the safe default; records
  ``harvest_not_yet_wired`` for every cell. Used by the test
  surface so unit tests never spawn subprocesses.
* ``live_tier2_harvest_fn`` — the opt-in live invocation;
  spawns ``sol-execbench`` as a subprocess with the NVBit
  metric extractor injected via ``CUDA_INJECTION64_PATH`` and
  the Tier-2 address-dump enabled. Callers wire it via
  ``harvest_pilot_cohort(harvest_fn=live_tier2_harvest_fn)`` OR
  via the module's CLI ``--live`` flag.

Sidecar path convention (matches the live Tier-2 harvester's
output): ``<slug>__<lang>/nvbit_metrics/tier2_address_trace.t2a``
under the harvest output root.

Output (per pilot cohort):

    [meta]
    schema_version
    host_arch
    captured_at_utc
    row_count
    aggregate_attempted
    aggregate_completed
    aggregate_skipped_existing
    aggregate_failed
    aggregate_not_yet_wired

    [[entries]]
    slug
    language
    sidecar_relpath                 # repo-relative or sandbox-relative
    sidecar_present_before
    sidecar_present_after
    attempted
    completed
    failure_reason                  # closed vocabulary or ""
    tier2_sidecar_sha256            # empty when not completed
    tier2_record_count              # NULL_INT sentinel when not completed
    tier2_launch_count              # NULL_INT sentinel when not completed
    tier2_byte_budget_used          # NULL_INT sentinel when not completed
    truncation_relaxation_applied   # false when not completed
    captured_at_utc
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Final


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "sm120_pilot_tier2_harvest_status_v1.toml"
)
_DEFAULT_HARVEST_OUTPUT_ROOT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / ".harness_traces"
)

HARVEST_STATUS_SCHEMA_VERSION: Final[str] = "1.1"
HARVEST_STATUS_HOST_ARCH: Final[str] = "sm_120"

# Per-cell Tier-2 sidecar byte budget cap. Sidecars whose
# parsed file size exceeds this cap are demoted to
# ``tier2_budget_exceeded`` even when they parse cleanly —
# the predictor pipeline cannot honestly drive a simulator
# over a trace fragment that overflowed the producer-side
# byte budget without the truncation-relaxation marker, and
# accepting an over-cap parseable sidecar would silently
# corrupt downstream MAPE attribution.
TIER2_SIDECAR_BUDGET_BYTES: Final[int] = 256 * 1024 * 1024

# Sentinels for NULL ints in per-cell sidecar per-row evidence; matches the
# diagnostics sidecar's NULL_INT convention so downstream readers
# don't need a separate vocabulary.
_NULL_INT: Final[int] = -1
_NULL_SHA: Final[str] = ""

# Closed vocabulary of failure-reason tokens. Empty string is
# the implicit "no failure" value when ``completed=True``.
FAILURE_REASON_VOCABULARY: Final[tuple[str, ...]] = (
    "harvest_not_yet_wired",
    "tier2_harvest_subprocess_failed",
    "tier2_harvest_timeout",
    "tier2_harvest_sidecar_empty",
    "tier2_harvest_sidecar_parse_failed",
    "tier2_budget_exceeded",
    "tier2_no_launches_captured",
)


@dataclass(frozen=True, slots=True)
class HarvestStatusRow:
    slug: str
    language: str
    sidecar_relpath: str
    sidecar_present_before: bool
    sidecar_present_after: bool
    attempted: bool
    completed: bool
    failure_reason: str
    # Per-cell sidecar evidence. Populated by parsing the
    # produced sidecar when ``completed=True``; NULL sentinels
    # (``""`` / ``-1`` / ``False``) when not completed.
    tier2_sidecar_sha256: str
    tier2_record_count: int
    tier2_launch_count: int
    tier2_byte_budget_used: int
    truncation_relaxation_applied: bool
    captured_at_utc: str


@dataclass(frozen=True, slots=True)
class HarvestOutcome:
    """Return type for a pluggable ``harvest_fn``.

    ``completed`` is True when the harvest produced a valid
    sidecar; ``failure_reason`` must be a member of
    ``FAILURE_REASON_VOCABULARY`` (or empty when ``completed``).
    """

    completed: bool
    failure_reason: str


class HarvestOutcomeInvalidError(ValueError):
    """Raised by ``harvest_pilot_cohort`` when the
    ``HarvestOutcome`` returned by ``harvest_fn`` violates the
    closed contract:

    - ``completed=True`` requires ``failure_reason == ""``.
    - ``completed=False`` requires ``failure_reason`` to be a
      member of ``FAILURE_REASON_VOCABULARY``.

    The completed-without-sidecar case is NOT an outcome
    violation — it is silently demoted to
    ``completed=False`` + ``failure_reason="tier2_harvest_sidecar_empty"``
    inside the driver (the harvest_fn promised a sidecar but
    the disk does not show one; treat as the typed failure
    rather than crash, so the batch can continue).
    """


def _validate_harvest_outcome(outcome: HarvestOutcome) -> None:
    """Raise ``HarvestOutcomeInvalidError`` when the outcome
    violates the closed contract.

    See ``HarvestOutcomeInvalidError`` for the full contract.
    Does NOT check sidecar existence — that's the driver's
    responsibility (the driver demotes silently rather than
    raises so a single bad cell doesn't abort the batch).
    """
    if outcome.completed and outcome.failure_reason:
        raise HarvestOutcomeInvalidError(
            f"completed=True requires failure_reason='', got "
            f"failure_reason={outcome.failure_reason!r}"
        )
    if not outcome.completed:
        if not outcome.failure_reason:
            raise HarvestOutcomeInvalidError(
                "completed=False requires a typed failure_reason; "
                "got empty string"
            )
        if outcome.failure_reason not in FAILURE_REASON_VOCABULARY:
            raise HarvestOutcomeInvalidError(
                f"failure_reason={outcome.failure_reason!r} not in "
                f"FAILURE_REASON_VOCABULARY "
                f"{sorted(FAILURE_REASON_VOCABULARY)}"
            )


def _stub_harvest_fn(
    *,
    slug: str,
    language: str,
    sidecar_path: Path,
) -> HarvestOutcome:
    """Default ``harvest_fn`` — records ``harvest_not_yet_wired``
    for every cell.

    The opt-in live wiring lives at ``live_tier2_harvest_fn``;
    callers that want the actual NVBit invocation pass that
    function explicitly to ``harvest_pilot_cohort``."""
    return HarvestOutcome(
        completed=False,
        failure_reason="harvest_not_yet_wired",
    )


# Paths for the live harvester. Resolved lazily so import of
# this module never touches disk (a deferred-import discipline
# the deployed predictor surface already follows).
_NVBIT_SO = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "metric_extractor.so"
)
_SOL_EXECBENCH_BIN = _REPO_ROOT / ".venv" / "bin" / "sol-execbench"
_BASELINES_ROOT = _REPO_ROOT / "baselines"
_BENCHMARK_ROOT = _REPO_ROOT / "sol-execbench" / "data" / "benchmark"

# Per-cell baseline category lookup the live harvester needs.
# (slug, language) -> (baselines_category, benchmark_subset).
# Same shape the reverification driver uses; pinned here so the
# live harvester can resolve paths without an external file.
_PILOT_PATH_LOOKUP: Final[dict[tuple[str, str], tuple[str, str]]] = {
    ("problem_024_rmsnorm_h2048", "cuda_cpp"): ("FIB", "FlashInfer-Bench"),
    ("problem_002_fused_add_rmsnorm_h4096", "cuda_cpp"): (
        "FIB", "FlashInfer-Bench",
    ),
    ("problem_069_rms_norm", "cuda_cpp"): ("L1", "L1"),
    ("problem_025_video_latent_gelu_activation", "cuda_cpp"): ("L1", "L1"),
    ("problem_068_gelu_approximate_feedforward_backward", "cuda_cpp"): (
        "L2", "L2",
    ),
    ("problem_045_fused_linear_gelu_grn_linear", "cuda_cpp"): ("L1", "L1"),
    (
        "problem_030_flux_concatenated_sequence_processing_with_split",
        "cuda_cpp",
    ): ("L2", "L2"),
    ("problem_010_attention_value_projection_with_transpose", "cuda_cpp"): (
        "L1", "L1",
    ),
    ("problem_063_attention_output_reshape_and_projection", "cuda_cpp"): (
        "L1", "L1",
    ),
    (
        "problem_051_attention_qkv_with_qk_norm_single_kernel_backward",
        "cuda_cpp",
    ): ("L1", "L1"),
    (
        "problem_021_cross_attention_text_video_conditioning_backward",
        "cuda_cpp",
    ): ("L2", "L2"),
    ("problem_093_grouped_topk_moe_routing_backward", "cuda_cpp"): (
        "L1", "L1",
    ),
    # ---- Cross-arch plan the calibration workstream (sm_86 5-cell pilot) ----
    # Additive entries for the 5 R19-R23 sm_86 admitted pilots.
    # Adding here does not enroll them into the sm_120 Tier-2
    # cohort because the Tier-2 driver also gates on the
    # cohort-specific kernel-name filter map.
    ("problem_006_hyena_depthwise_conv1d_split_gate", "cuda_cpp"): (
        "L1", "L1",
    ),
    ("problem_040_conv2d_residual_block", "cuda_cpp"): (
        "L1", "L1",
    ),
    ("problem_066_resnet_block_with_time_embedding", "cuda_cpp"): (
        "L2", "L2",
    ),
    (
        "problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection",
        "cuda_cpp",
    ): ("L1", "L1"),
    ("problem_087_embedding_with_initial_layernorm_backward", "cuda_cpp"): (
        "L1", "L1",
    ),
}


def _benchmark_def_name(slug: str) -> str:
    """Convert ``problem_<defn>`` to ``<defn>`` (the sol-execbench
    benchmark dir basename)."""
    if slug.startswith("problem_"):
        return slug[len("problem_"):]
    return slug


def live_tier2_harvest_fn(
    *,
    slug: str,
    language: str,
    sidecar_path: Path,
    tier2_budget_bytes: int = 256 * 1024 * 1024,
    timeout_s: int = 600,
) -> HarvestOutcome:
    """Opt-in live Tier-2 NVBit harvester for a pilot cell.

    Invokes ``sol-execbench`` as a subprocess with the
    NVBit metric extractor injected via ``CUDA_INJECTION64_PATH``
    and the Tier-2 address-dump enabled. The produced sidecar
    lands at ``sidecar_path.parent / "tier2_address_trace.t2a"``
    (which IS ``sidecar_path`` per the driver's path convention).

    Pre-conditions checked at call time:
      - ``(slug, language)`` is in ``_PILOT_PATH_LOOKUP``.
      - The NVBit metric extractor .so is built and exists at
        ``_NVBIT_SO``.
      - The pilot cell's ``solution.json`` exists on disk.
      - The benchmark problem dir exists on disk.

    Any pre-condition failure returns a typed
    ``tier2_harvest_subprocess_failed`` outcome.

    The harvest is opt-in: pass this function explicitly to
    ``harvest_pilot_cohort(harvest_fn=live_tier2_harvest_fn)``
    OR via the module's CLI ``--live`` flag. The default
    ``harvest_fn`` remains the safe stub.
    """
    import os
    import subprocess as _subprocess

    if (slug, language) not in _PILOT_PATH_LOOKUP:
        return HarvestOutcome(
            completed=False,
            failure_reason="tier2_harvest_subprocess_failed",
        )
    baselines_category, benchmark_subset = _PILOT_PATH_LOOKUP[
        (slug, language)
    ]
    solution_path = (
        _BASELINES_ROOT
        / baselines_category
        / slug
        / language
        / "solution.json"
    )
    benchmark_dir = (
        _BENCHMARK_ROOT
        / benchmark_subset
        / _benchmark_def_name(slug)
    )
    if not solution_path.is_file():
        return HarvestOutcome(
            completed=False,
            failure_reason="tier2_harvest_subprocess_failed",
        )
    if not benchmark_dir.is_dir():
        return HarvestOutcome(
            completed=False,
            failure_reason="tier2_harvest_subprocess_failed",
        )
    if not _NVBIT_SO.is_file():
        return HarvestOutcome(
            completed=False,
            failure_reason="tier2_harvest_subprocess_failed",
        )

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_INJECTION64_PATH"] = str(_NVBIT_SO)
    env["METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP"] = "1"
    env["METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES"] = str(
        tier2_budget_bytes
    )
    env["METRIC_EXTRACTOR_LAUNCHES_PATH"] = str(
        sidecar_path.parent / "launches.jsonl"
    )
    # Permissive user-module filter: sol-execbench-launched
    # extensions load compiled CUTLASS .so files whose names do
    # NOT match the default ``.*sol_execbench.*`` regex.
    # ``.*`` admits every loaded module so the per-launch metric
    # extractor sees the pilot kernel's launches.
    env["METRIC_EXTRACTOR_USER_MODULE_REGEX"] = ".*"
    # Higher slot budget so multi-launch pilots emit every
    # qualifying kernel launch rather than just the first.
    env["METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS"] = "32"
    env["CUTLASS_DIR"] = str(_REPO_ROOT / "third_party" / "cutlass")
    # Per-cell kernel-name filter: with Tier-2 address
    # dump on, the per-slot Tier-2 buffer is ~503 MiB and the
    # default 512 MiB per-process budget admits only ONE accepted
    # launch. PyTorch's curand RNG kernel initialises input
    # tensors BEFORE the pilot kernel runs; without the kernel-
    # name filter that curand launch grabs the single slot and
    # the pilot's 280 launches all skip with
    # ``per_process_budget_exceeded``. The producer-side regex
    # filter (added in an earlier iteration alongside ``USER_MODULE_REGEX``)
    # rejects non-matching launches before slot allocation, so
    # the slot pool stays free for the target pilot kernel.
    # ATen-only pilots (whose work runs through cuBLAS / cuDNN
    # with shape-dependent kernel selection) carry no entry in
    # PILOT_KERNEL_NAME_BASE; for those cells we omit the env
    # var so the harvester behaves as before — those cells are
    # excluded from the predictor ledger via the design contract's
    # ``ncu_kernel_name_not_pinned`` token, NOT via Tier-2
    # admission.
    from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.pilot_kernel_filter import (  # noqa: E501
        lookup_pilot_kernel_name_base,
    )
    kernel_name_base = lookup_pilot_kernel_name_base(slug, language)
    if kernel_name_base is not None:
        env["METRIC_EXTRACTOR_KERNEL_NAME_REGEX"] = kernel_name_base

    argv = [
        str(_SOL_EXECBENCH_BIN),
        str(benchmark_dir),
        "--solution",
        str(solution_path),
        "--json",
        "--compile-timeout",
        str(timeout_s),
        "--timeout",
        str(timeout_s),
    ]
    try:
        result = _subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s * 2,
            env=env,
        )
    except _subprocess.TimeoutExpired:
        return HarvestOutcome(
            completed=False, failure_reason="tier2_harvest_timeout",
        )

    # The sidecar is the source of truth, NOT the subprocess
    # exit code. NVBit injection slows the kernel enough that
    # sol-execbench's correctness / latency check can fail
    # AFTER the metric extractor has already written a complete
    # valid sidecar. Check the sidecar first.
    if (
        sidecar_path.exists()
        and sidecar_path.stat().st_size > 0
    ):
        return HarvestOutcome(completed=True, failure_reason="")
    # No sidecar produced. Disambiguate: a non-zero subprocess
    # exit with no sidecar means the harvester didn't reach the
    # kernel launch (compile error, missing dependency, etc.);
    # a zero exit with no sidecar means the user-module regex
    # filtered every load OR no qualifying launches happened.
    if result.returncode != 0:
        return HarvestOutcome(
            completed=False,
            failure_reason="tier2_harvest_subprocess_failed",
        )
    return HarvestOutcome(
        completed=False,
        failure_reason="tier2_no_launches_captured",
    )


def _sidecar_path_for_cell(
    *,
    slug: str,
    language: str,
    harvest_output_root: Path,
) -> Path:
    """Per-cell expected Tier-2 sidecar path under the harvest
    output root.

    Aligned with the existing live harvester
    (``sm120_l2_associativity_probe/harvest_tier2_traces.py``)
    which writes
    ``<cell>/nvbit_metrics/tier2_address_trace.t2a``. The
    coverage-expansion driver uses
    ``<slug>__<lang>/nvbit_metrics/tier2_address_trace.t2a`` so
    the existing harvester's output drops directly into the
    per-cell bundle a downstream resume / SHA / diagnostics
    consumer expects. The bundle name follows the
    ``<slug>__<lang>`` convention the harness traces tree uses
    for everything else; pilot variants are keyed by
    (slug, language) so multi-language pilots get separate
    bundles without collision.
    """
    bundle = harvest_output_root / f"{slug}__{language}"
    return bundle / "nvbit_metrics" / "tier2_address_trace.t2a"


def _sidecar_is_valid(path: Path) -> bool:
    """A sidecar is "valid" for resume purposes if it exists
    and is non-empty. Empty sidecars are treated as a previous
    failed run and re-attempted."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class _SidecarSummary:
    """per-cell sidecar evidence extracted from a successful sidecar."""

    sha256: str
    record_count: int
    launch_count: int
    byte_budget_used: int
    truncation_relaxation_applied: bool
    parse_failed: bool


def _summarize_sidecar(sidecar_path: Path) -> _SidecarSummary:
    """Parse the produced sidecar and surface the per-cell sidecar evidence
    fields the status emitter records per row.

    Returns a ``_SidecarSummary`` with the file's SHA-256 and
    parsed-frame totals on success. If the parser raises (the
    file was produced but is malformed / truncated mid-frame /
    schema-violating), returns ``parse_failed=True`` so the
    caller can demote the row to the typed
    ``tier2_harvest_sidecar_parse_failed`` reason.

    The on-disk file size is the authoritative
    ``byte_budget_used`` regardless of parser fidelity (the
    producer guarantees ``file_size <= TIER2_TRACE_BUDGET_BYTES``).
    Frame-level ``budget_bytes_used`` mirrors are NOT summed
    because they overcount when launches reuse the streaming
    buffer.
    """
    file_bytes = sidecar_path.read_bytes()
    sha = hashlib.sha256(file_bytes).hexdigest()
    byte_budget_used = len(file_bytes)
    # Defer import: tier2_sidecar_parser pulls in struct + dataclasses;
    # keeping it off the module-level import keeps the harvest driver
    # cheap to import in tests that don't touch real sidecars.
    from sol_baselines_harness.metrics import tier2_sidecar_parser

    try:
        frames = tier2_sidecar_parser.parse_launch_frames(sidecar_path)
    except tier2_sidecar_parser.Tier2ParserError:
        return _SidecarSummary(
            sha256=sha,
            record_count=_NULL_INT,
            launch_count=_NULL_INT,
            byte_budget_used=byte_budget_used,
            truncation_relaxation_applied=False,
            parse_failed=True,
        )
    record_count = sum(f.access_record_count for f in frames)
    launch_count = len(frames)
    truncation_relaxation_applied = any(
        f.truncation is not None for f in frames
    )
    return _SidecarSummary(
        sha256=sha,
        record_count=record_count,
        launch_count=launch_count,
        byte_budget_used=byte_budget_used,
        truncation_relaxation_applied=truncation_relaxation_applied,
        parse_failed=False,
    )


def harvest_pilot_cohort(
    pilot_cells: list[tuple[str, str]],
    *,
    harvest_output_root: Path = _DEFAULT_HARVEST_OUTPUT_ROOT,
    harvest_fn: Callable[..., HarvestOutcome] = _stub_harvest_fn,
    captured_at_utc: str | None = None,
    max_sidecar_bytes: int = TIER2_SIDECAR_BUDGET_BYTES,
) -> list[HarvestStatusRow]:
    """Drive the Tier-2 harvest over a pilot cohort.

    Per-cell semantics:

    1. **Resume**: if a non-empty sidecar exists at the
       expected path, skip the harvest. Record
       ``sidecar_present_before=True``,
       ``attempted=False``, ``completed=True``,
       ``failure_reason=""``. per-cell sidecar evidence fields are
       extracted from the existing sidecar.
    2. **Harvest**: invoke ``harvest_fn`` with the per-cell
       paths. The harvester is responsible for actually
       writing the sidecar; this driver records the outcome
       AND parses the produced sidecar to extract the
       per-cell per-cell sidecar evidence (SHA-256, record count, launch
       count, byte budget used, truncation-relaxation flag).
    3. **Failure logging**: when the harvest returns
       ``completed=False``, the row records the typed failure
       reason from the closed vocabulary, and per-cell sidecar evidence
       fields fall back to NULL sentinels. The driver
       CONTINUES to the next cell (no batch-level abort).
    4. **Parser-failure demotion**: a sidecar that exists +
       hashes cleanly but whose binary layout violates the
       parser's schema invariants is demoted to
       ``failure_reason="tier2_harvest_sidecar_parse_failed"``
       so downstream consumers know the file is unusable.

    Args:
        pilot_cells: list of (slug, language) tuples.
        harvest_output_root: directory under which the per-cell
            ``<slug>__<lang>/nvbit_metrics/tier2_address_trace.t2a``
            sidecars land (matches the live Tier-2 harvester's
            output convention).
        harvest_fn: pluggable harvester. Defaults to
            ``_stub_harvest_fn`` (safe; records
            ``harvest_not_yet_wired``); pass
            ``live_tier2_harvest_fn`` to opt into the real
            NVBit-injected sol-execbench invocation.
        captured_at_utc: optional ISO-8601 UTC timestamp to
            stamp on every row's ``captured_at_utc``. When None,
            each row gets its own per-call timestamp (slightly
            different across cells); supply an explicit value
            for byte-stable artifact emission.

    Returns:
        List of ``HarvestStatusRow`` sorted by (slug, language).
    """
    rows: list[HarvestStatusRow] = []
    for slug, language in pilot_cells:
        sidecar_path = _sidecar_path_for_cell(
            slug=slug,
            language=language,
            harvest_output_root=harvest_output_root,
        )
        sidecar_present_before = _sidecar_is_valid(sidecar_path)
        timestamp = captured_at_utc or datetime.now(
            timezone.utc
        ).replace(microsecond=0).isoformat()

        if sidecar_present_before:
            summary = _summarize_sidecar(sidecar_path)
            if summary.parse_failed:
                rows.append(
                    _build_failure_row(
                        slug=slug,
                        language=language,
                        sidecar_path=sidecar_path,
                        sidecar_present_before=True,
                        sidecar_present_after=True,
                        attempted=False,
                        failure_reason=(
                            "tier2_harvest_sidecar_parse_failed"
                        ),
                        timestamp=timestamp,
                    )
                )
                continue
            if summary.byte_budget_used > max_sidecar_bytes:
                rows.append(
                    _build_failure_row(
                        slug=slug,
                        language=language,
                        sidecar_path=sidecar_path,
                        sidecar_present_before=True,
                        sidecar_present_after=True,
                        attempted=False,
                        failure_reason="tier2_budget_exceeded",
                        timestamp=timestamp,
                    )
                )
                continue
            rows.append(
                HarvestStatusRow(
                    slug=slug,
                    language=language,
                    sidecar_relpath=_to_repo_relative(sidecar_path),
                    sidecar_present_before=True,
                    sidecar_present_after=True,
                    attempted=False,
                    completed=True,
                    failure_reason="",
                    tier2_sidecar_sha256=summary.sha256,
                    tier2_record_count=summary.record_count,
                    tier2_launch_count=summary.launch_count,
                    tier2_byte_budget_used=summary.byte_budget_used,
                    truncation_relaxation_applied=(
                        summary.truncation_relaxation_applied
                    ),
                    captured_at_utc=timestamp,
                )
            )
            continue

        try:
            outcome = harvest_fn(
                slug=slug,
                language=language,
                sidecar_path=sidecar_path,
            )
        except Exception as exc:  # noqa: BLE001
            outcome = HarvestOutcome(
                completed=False,
                failure_reason="tier2_harvest_subprocess_failed",
            )
            print(
                f"  {slug} ({language}): harvest raised "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        # Validate outcome against the closed contract. Bad
        # outcomes raise (HarvestOutcomeInvalidError); they
        # represent a programmer / wiring error in the
        # harvest_fn, not a per-cell harvest failure.
        _validate_harvest_outcome(outcome)
        sidecar_present_after = _sidecar_is_valid(sidecar_path)
        # Completed-implies-sidecar enforcement: a harvest_fn
        # that returns completed=True but did NOT actually
        # write a non-empty sidecar gets demoted to
        # completed=False + failure_reason=tier2_harvest_sidecar_empty.
        # The driver demotes (rather than raises) so the batch
        # continues to the next cell.
        if outcome.completed and not sidecar_present_after:
            outcome = HarvestOutcome(
                completed=False,
                failure_reason="tier2_harvest_sidecar_empty",
            )

        if outcome.completed:
            summary = _summarize_sidecar(sidecar_path)
            if summary.parse_failed:
                rows.append(
                    _build_failure_row(
                        slug=slug,
                        language=language,
                        sidecar_path=sidecar_path,
                        sidecar_present_before=False,
                        sidecar_present_after=sidecar_present_after,
                        attempted=True,
                        failure_reason=(
                            "tier2_harvest_sidecar_parse_failed"
                        ),
                        timestamp=timestamp,
                    )
                )
                continue
            if summary.byte_budget_used > max_sidecar_bytes:
                rows.append(
                    _build_failure_row(
                        slug=slug,
                        language=language,
                        sidecar_path=sidecar_path,
                        sidecar_present_before=False,
                        sidecar_present_after=sidecar_present_after,
                        attempted=True,
                        failure_reason="tier2_budget_exceeded",
                        timestamp=timestamp,
                    )
                )
                continue
            rows.append(
                HarvestStatusRow(
                    slug=slug,
                    language=language,
                    sidecar_relpath=_to_repo_relative(sidecar_path),
                    sidecar_present_before=False,
                    sidecar_present_after=sidecar_present_after,
                    attempted=True,
                    completed=True,
                    failure_reason="",
                    tier2_sidecar_sha256=summary.sha256,
                    tier2_record_count=summary.record_count,
                    tier2_launch_count=summary.launch_count,
                    tier2_byte_budget_used=summary.byte_budget_used,
                    truncation_relaxation_applied=(
                        summary.truncation_relaxation_applied
                    ),
                    captured_at_utc=timestamp,
                )
            )
        else:
            rows.append(
                _build_failure_row(
                    slug=slug,
                    language=language,
                    sidecar_path=sidecar_path,
                    sidecar_present_before=False,
                    sidecar_present_after=sidecar_present_after,
                    attempted=True,
                    failure_reason=outcome.failure_reason,
                    timestamp=timestamp,
                )
            )
    rows.sort(key=lambda r: (r.slug, r.language))
    return rows


def _build_failure_row(
    *,
    slug: str,
    language: str,
    sidecar_path: Path,
    sidecar_present_before: bool,
    sidecar_present_after: bool,
    attempted: bool,
    failure_reason: str,
    timestamp: str,
) -> HarvestStatusRow:
    """Construct a HarvestStatusRow with NULL per-cell sidecar sentinels for
    a failed / unattempted / parse-failed cell."""
    return HarvestStatusRow(
        slug=slug,
        language=language,
        sidecar_relpath=_to_repo_relative(sidecar_path),
        sidecar_present_before=sidecar_present_before,
        sidecar_present_after=sidecar_present_after,
        attempted=attempted,
        completed=False,
        failure_reason=failure_reason,
        tier2_sidecar_sha256=_NULL_SHA,
        tier2_record_count=_NULL_INT,
        tier2_launch_count=_NULL_INT,
        tier2_byte_budget_used=_NULL_INT,
        truncation_relaxation_applied=False,
        captured_at_utc=timestamp,
    )


def _to_repo_relative(path: Path) -> str:
    """Convert an absolute path under the repo to a repo-relative
    string; pass through any path outside the repo."""
    try:
        return str(path.resolve().relative_to(_REPO_ROOT.resolve()))
    except (ValueError, OSError):
        return str(path)


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_harvest_status_toml(
    *,
    captured_at_utc: str,
    rows: list[HarvestStatusRow],
) -> str:
    """Render the harvest status artifact TOML."""
    aggregate_attempted = sum(1 for r in rows if r.attempted)
    aggregate_completed = sum(1 for r in rows if r.completed)
    aggregate_skipped_existing = sum(
        1 for r in rows if r.sidecar_present_before
    )
    aggregate_failed = sum(
        1 for r in rows if r.attempted and not r.completed
    )
    aggregate_not_yet_wired = sum(
        1 for r in rows if r.failure_reason == "harvest_not_yet_wired"
    )
    payload_sha = hashlib.sha256(
        "\n".join(
            f"{r.slug}|{r.language}|{int(r.completed)}|"
            f"{r.failure_reason}"
            for r in rows
        ).encode("utf-8")
    ).hexdigest()
    lines: list[str] = []
    lines.append("# sm_120 pilot Tier-2 harvest status artifact.")
    lines.append("#")
    lines.append(
        "# Per-cell outcomes from the Tier-2 NVBit harvest"
    )
    lines.append(
        "# pipeline against the pilot cohort. Resume semantics:"
    )
    lines.append(
        "# cells with an existing non-empty sidecar are skipped"
    )
    lines.append(
        "# (sidecar_present_before=true, attempted=false,"
    )
    lines.append("# completed=true).")
    lines.append("")
    lines.append("[meta]")
    lines.append(
        f'schema_version = "{HARVEST_STATUS_SCHEMA_VERSION}"'
    )
    lines.append(f'host_arch = "{HARVEST_STATUS_HOST_ARCH}"')
    lines.append(f'captured_at_utc = "{captured_at_utc}"')
    lines.append(f"row_count = {len(rows)}")
    lines.append(f"aggregate_attempted = {aggregate_attempted}")
    lines.append(f"aggregate_completed = {aggregate_completed}")
    lines.append(
        f"aggregate_skipped_existing = {aggregate_skipped_existing}"
    )
    lines.append(f"aggregate_failed = {aggregate_failed}")
    lines.append(
        f"aggregate_not_yet_wired = {aggregate_not_yet_wired}"
    )
    lines.append(f'payload_sha256 = "{payload_sha}"')
    lines.append("")
    for row in rows:
        lines.append("[[entries]]")
        lines.append(f'slug = "{_toml_escape(row.slug)}"')
        lines.append(f'language = "{_toml_escape(row.language)}"')
        lines.append(
            f'sidecar_relpath = "{_toml_escape(row.sidecar_relpath)}"'
        )
        lines.append(
            f"sidecar_present_before = "
            f"{str(row.sidecar_present_before).lower()}"
        )
        lines.append(
            f"sidecar_present_after = "
            f"{str(row.sidecar_present_after).lower()}"
        )
        lines.append(f"attempted = {str(row.attempted).lower()}")
        lines.append(f"completed = {str(row.completed).lower()}")
        lines.append(
            f'failure_reason = "{_toml_escape(row.failure_reason)}"'
        )
        lines.append(
            f'tier2_sidecar_sha256 = "'
            f'{_toml_escape(row.tier2_sidecar_sha256)}"'
        )
        lines.append(f"tier2_record_count = {row.tier2_record_count}")
        lines.append(f"tier2_launch_count = {row.tier2_launch_count}")
        lines.append(
            f"tier2_byte_budget_used = {row.tier2_byte_budget_used}"
        )
        lines.append(
            f"truncation_relaxation_applied = "
            f"{str(row.truncation_relaxation_applied).lower()}"
        )
        lines.append(
            f'captured_at_utc = "{row.captured_at_utc}"'
        )
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def emit_harvest_status(
    pilot_cells: list[tuple[str, str]],
    *,
    output: Path,
    captured_at_utc: str,
    harvest_output_root: Path = _DEFAULT_HARVEST_OUTPUT_ROOT,
    harvest_fn: Callable[..., HarvestOutcome] = _stub_harvest_fn,
    max_sidecar_bytes: int = TIER2_SIDECAR_BUDGET_BYTES,
) -> Path:
    rows = harvest_pilot_cohort(
        pilot_cells,
        harvest_output_root=harvest_output_root,
        harvest_fn=harvest_fn,
        captured_at_utc=captured_at_utc,
        max_sidecar_bytes=max_sidecar_bytes,
    )
    toml_text = render_harvest_status_toml(
        captured_at_utc=captured_at_utc, rows=rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml_text, encoding="utf-8")
    return output
