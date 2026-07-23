"""Microbenchmark-suite coverage inventory.

The original-plan microbenchmark-suite criterion enumerates
canonical kernels that the data-collection toolchain must
exercise to validate cache-model behaviors:

1. Pointer chase for latency bands.
2. Working-set sweep for capacity knees.
3. Stride/misalignment sweep for sector amplification.
4. Warm/cold cache pair.
5. `.ca` / `.cg` / `.cs` cache-modifier variants.
6. L2 prefetch hint variants where supported.
7. L2 persisting-window sweep where supported.
8. Carveout sensitivity probe.
9. Bandwidth saturation sweep.
10. Resident-warp pressure sweep.

This module declares the inventory as a typed tuple of
`Ac14Family` records. Each family carries a `state` drawn from
a closed enum:

- `live_verified_sm_120` — sm_120 live evidence captured at a
  reviewed HEAD; the slug's trace directory inherits the HEAD
  identity per the abstract-HEAD pattern.
- `structural_only` — slug + shape file exist; no live trace at
  the round-closing HEAD yet. CLAUDE.md "Two-Device Local
  Development Model" allows this evidence status under the
  rule that any AC sub-criterion declared closed must queue a
  `Verified-live-<arch>` follow-up.
- `arch_unsupported` — typed-unavailable on the active arch
  (e.g., a feature gate failed); the family ships a typed
  rationale rather than silent omission.
- `not_implemented` — no slug yet exists; the family must be
  closed before the suite-coverage criterion can verify.

The companion offline validator
(`tools/test_microbench_ac14_coverage.py`) regression-locks the
contract: every family is in exactly one state; every
`live_verified_sm_120` or `structural_only` slug must exist in
at least one of the simbench-or-portable_lcd rosters and have
a shape file with the three required shape blocks
(`timing_shapes`, `trace_shapes`, `correctness_shapes`); the
validator FAILS when any family is `not_implemented`.

R43 review rejected the prior "covered XOR pending" model and
the deferral-as-coverage premise; R44 replaces the model with
the four-state form so the validator cannot pass while the
suite is incomplete.

Filename `microbench_inventory.py` is unique across
`tools/<subdir>/` per `BL-20260505-tools-subdir-module-name-collision`
(no other module of this basename exists under `tools/`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Ac14FamilyState(str, Enum):
    """Closed enum of family states. The validator enforces that
    `not_implemented` causes a hard failure; the other states are
    accepted under the disciplines documented in the module
    docstring."""

    live_verified_sm_120 = "live_verified_sm_120"
    structural_only = "structural_only"
    arch_unsupported = "arch_unsupported"
    not_implemented = "not_implemented"


@dataclass(frozen=True)
class Ac14SlugCoverage:
    """One simbench-or-portable_lcd roster slug that covers a
    family.

    `slug` is the roster row's `slug` field. `roster` names the
    workload registry the row lives in (`simbench` or
    `portable_lcd`). `rationale` is a one-line explanation of how
    the slug exercises the family.
    """

    slug: str
    roster: str
    rationale: str


@dataclass(frozen=True)
class Ac14Family:
    """One row in the suite-coverage criterion's family inventory.

    The tuple of these forms the closed contract surface. Each
    family declares a `state`, which determines which fields the
    validator inspects:

    - `live_verified_sm_120`: requires `covered_slugs` non-empty;
      every slug must exist in at least one roster and have a
      shape file with the three required blocks.
    - `structural_only`: same slug-existence checks; live trace
      not yet captured (the validator records this as a residual
      gap).
    - `arch_unsupported`: requires `arch_unsupported_reason`
      non-empty; no slug check.
    - `not_implemented`: requires `not_implemented_reason`
      non-empty; the validator FAILS on any family in this
      state.
    """

    family_id: str
    plan_description: str
    state: Ac14FamilyState
    covered_slugs: tuple[Ac14SlugCoverage, ...] = field(default_factory=tuple)
    arch_unsupported_reason: str = ""
    not_implemented_reason: str = ""


# The plan-enumerated families. Order matches the original-plan
# enumeration. The 5th plan family (`.ca`/`.cg`/`.cs` cache-
# modifier variants) is recorded as one inventory row; the per-
# variant SASS evidence surface lives in the perturbation-harness
# clause's typed-reason set (e.g., cache_modifier_ca,
# cache_modifier_cg, cache_modifier_cs). Adding or removing a
# family is a contract change requiring an explicit plan update;
# the validator's exact-count assertion catches accidental drift.
_AC14_FAMILIES: tuple[Ac14Family, ...] = (
    Ac14Family(
        family_id="pointer_chase",
        plan_description="Pointer chase for latency bands.",
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="l2_thrash_pointer_chase",
                roster="simbench",
                rationale=(
                    "Each warp follows a long dependent-load chain "
                    "exceeding L2 capacity to drive L2 miss rate; "
                    "shape `chain_length` axis sweeps the latency "
                    "band. R53 captured canonical live sm_120 "
                    "trace + per-shape sidecars on the RTX 5060 "
                    "Laptop GPU via `run_suite.py --mode trace`: "
                    "workloads/simbench_bw/results/"
                    "l2_thrash_pointer_chase/blackwell_consumer_unknown/"
                    "sm_120/ admits via verify_artifacts.py and "
                    "contains trace_trace_00.json + "
                    "measurement_manifest.json + "
                    "trace_trace_00/dynamic_trace.pb (NVBit dynamic "
                    "trace) + sector_stream/, hrd_profile/, "
                    "divergence_proxies/, sharing_locality/, "
                    "order_projections/, perturbation_run_ledger/, "
                    "convergence_confidence/ feature sidecars. "
                    "The R50 raw trees under "
                    "workloads/simbench_bw/l1/<slug>/traces/ "
                    "remain as historical evidence. The artifact "
                    "tree rebases on every commit per the "
                    "abstract-HEAD pattern."
                    "Live sm_86 cross-arch evidence captured on the RTX "
                    "3090 Ti (Ampere desktop, GA102, CC 8.6, NVBit 1.8, "
                    "CUDA 13.2, native g++-14) under "
                    "workloads/simbench_bw/results/l2_thrash_pointer_chase/rtx_3090_ti/sm_86/ "
                    "via the same `run_suite.py --mode trace` path "
                    "post-arch_gate widening (the slug's `arch_gate` was "
                    "widened to admit sm_86 in `roster.toml`, `Makefile`, "
                    "and the `.cu` source comment); admits via "
                    "`verify_artifacts.py` and carries the per-shape "
                    "feature sidecars (sector_stream, hrd_profile, "
                    "divergence_proxies, sharing_locality, "
                    "order_projections); the typed-fail-closed arms "
                    "(e.g., `hrd_layer_unavailable` when the upstream HRD "
                    "lacks usable admitted bins for one or both layers) "
                    "fire when triggered per the per-arch L1/L2 sizing "
                    "differences. "
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="working_set_sweep",
        plan_description="Working-set sweep for capacity knees.",
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="working_set_capacity_sweep",
                roster="simbench",
                rationale=(
                    "Pure-shape working-set sweep landed in R44: "
                    "each launch streams through a fresh "
                    "allocation sized by `working_set_bytes` so "
                    "successive shapes cross the L1 / L2 / DRAM "
                    "capacity knees on sm_120. The shape file "
                    "exposes `working_set_bytes` and `accesses` "
                    "axes; the kernel reads each entry once with "
                    "an opaque accumulator so the compiler cannot "
                    "elide the load. R53 captured canonical live "
                    "sm_120 trace + per-shape sidecars on the RTX "
                    "5060 Laptop GPU via `run_suite.py --mode "
                    "trace`: workloads/simbench_bw/results/"
                    "working_set_capacity_sweep/blackwell_consumer_unknown/"
                    "sm_120/ admits via verify_artifacts.py "
                    "(trace_trace_00.json + manifest + dynamic "
                    "trace + the per-shape feature sidecars (sector_stream, hrd_profile, divergence_proxies, sharing_locality, order_projections, perturbation_run_ledger, convergence_confidence)). "
                    "The artifact tree rebases on every commit "
                    "per the abstract-HEAD pattern."
                    "Live sm_86 cross-arch evidence captured on the RTX "
                    "3090 Ti (Ampere desktop, GA102, CC 8.6, NVBit 1.8, "
                    "CUDA 13.2, native g++-14) under "
                    "workloads/simbench_bw/results/working_set_capacity_sweep/rtx_3090_ti/sm_86/ "
                    "via the same `run_suite.py --mode trace` path "
                    "post-arch_gate widening (the slug's `arch_gate` was "
                    "widened to admit sm_86 in `roster.toml`, `Makefile`, "
                    "and the `.cu` source comment); admits via "
                    "`verify_artifacts.py` and carries the per-shape "
                    "feature sidecars (sector_stream, hrd_profile, "
                    "divergence_proxies, sharing_locality, "
                    "order_projections); the typed-fail-closed arms "
                    "(e.g., `hrd_layer_unavailable` when the upstream HRD "
                    "lacks usable admitted bins for one or both layers) "
                    "fire when triggered per the per-arch L1/L2 sizing "
                    "differences. "
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="stride_misalignment_sweep",
        plan_description=(
            "Stride/misalignment sweep for sector amplification."
        ),
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="dram_stream_stride_mix",
                roster="simbench",
                rationale=(
                    "Per-warp access stride mix exercises sector "
                    "amplification. Shape `stride_classes` axis "
                    "sweeps misalignment classes per warp; the "
                    "sector-stream generator already consumes this "
                    "slug's NVBit trace and emits the amplification "
                    "metric per warp instruction. Live sm_120 "
                    "trace exists under "
                    "workloads/simbench_bw/l1/dram_stream_stride_mix"
                    "/traces/."
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="warm_cold_pair",
        plan_description="Warm/cold cache pair.",
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="warm_cold_pair_relaunch",
                roster="simbench",
                rationale=(
                    "Paired-launch kernel landed in R44. The "
                    "trace_shape executes three launches over the "
                    "same allocation: a cold first-touch "
                    "(`warm_cold_pair_kernel`, allocation freshly "
                    "initialized), a between-pair eviction kernel "
                    "(`warm_cold_pair_eviction_kernel`, sized via "
                    "the `eviction_bytes` shape axis to flush the "
                    "L1/L2 working set), and a warm re-touch "
                    "(`warm_cold_pair_kernel` again on the same "
                    "allocation). The HRD-profiler criterion "
                    "compares the cold and warm launches' "
                    "per-launch hit-rate distributions to recover "
                    "the warm/cold delta; the eviction kernel's "
                    "trace is observed but not paired. R53 "
                    "captured canonical live sm_120 trace + the "
                    "full per-shape feature sidecar set (sector_"
                    "stream, hrd_profile, divergence_proxies, "
                    "sharing_locality, order_projections, "
                    "perturbation_run_ledger, convergence_"
                    "confidence) on the RTX 5060 Laptop GPU via "
                    "`run_suite.py --mode trace`: "
                    "workloads/simbench_bw/results/"
                    "warm_cold_pair_relaunch/blackwell_consumer_unknown/"
                    "sm_120/ admits via verify_artifacts.py. The "
                    "R53 closure was unblocked by replacing the "
                    "pre-R52 O(N^2) reuse-distance loop in "
                    "`tools/hrd_profiler/observation_builder.py` "
                    "and the duplicate O(N^2) loop in "
                    "`tools/order_projections/projector.py::"
                    "_hit_rate_proxy_under_order` with a Fenwick-"
                    "tree-backed `ReuseDistanceTracker` "
                    "(O(log N) per access, total O(N log N) per "
                    "launch). Pre-R53 the canonical capture "
                    "hung CPU-bound at 99.9% for 40+ minutes on "
                    "this slug because the multi-launch trace "
                    "(cold + eviction + warm — 3 kernels) "
                    "expands to a 6.4 MB sector_stream.jsonl. "
                    "The artifact tree rebases on every commit "
                    "per the abstract-HEAD pattern."
                    "Live sm_86 cross-arch evidence captured on the RTX "
                    "3090 Ti (Ampere desktop, GA102, CC 8.6, NVBit 1.8, "
                    "CUDA 13.2, native g++-14) under "
                    "workloads/simbench_bw/results/warm_cold_pair_relaunch/rtx_3090_ti/sm_86/ "
                    "via the same `run_suite.py --mode trace` path "
                    "post-arch_gate widening (the slug's `arch_gate` was "
                    "widened to admit sm_86 in `roster.toml`, `Makefile`, "
                    "and the `.cu` source comment); admits via "
                    "`verify_artifacts.py` and carries the per-shape "
                    "feature sidecars (sector_stream, hrd_profile, "
                    "divergence_proxies, sharing_locality, "
                    "order_projections); the typed-fail-closed arms "
                    "(e.g., `hrd_layer_unavailable` when the upstream HRD "
                    "lacks usable admitted bins for one or both layers) "
                    "fire when triggered per the per-arch L1/L2 sizing "
                    "differences. "
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="cache_modifier_variants",
        plan_description=(
            "`.ca` / `.cg` / `.cs` cache-modifier variants."
        ),
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="dram_stream_stride_mix",
                roster="simbench",
                rationale=(
                    "The perturbation-harness clause compiles all "
                    "three cache-modifier variants (`.ca`, `.cg`, "
                    "`.cs`) of this kernel via "
                    "tools/perturbation_harness/build.py and "
                    "emits admitted records under the "
                    "cache_modifier_ca / cache_modifier_cg / "
                    "cache_modifier_cs families. R45 added the "
                    "`.cs` (cache-streaming) variant; the SASS "
                    "confirmer admits per-arch encoding "
                    "differences (Ampere `LDG.E.CS` and Blackwell "
                    "`LDG.E.EF` per "
                    "BL-20260505-cross-arch-sass-mnemonic-divergence). "
                    "Live sm_120 evidence exists in the "
                    "perturbation_run_ledger.json artifact at "
                    "the reviewed HEAD."
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="l2_prefetch_hint_variants",
        plan_description=(
            "L2 prefetch hint variants where supported."
        ),
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="l2_prefetch_hint_sweep",
                roster="simbench",
                rationale=(
                    "R46 landing: each warp issues "
                    "`prefetch.global.L2 [addr]` PTX inline asm "
                    "`prefetch_distance` slots ahead of each read "
                    "in a sequential stream. The shape file's "
                    "`prefetch_distance` axis sweeps the hint "
                    "lookahead so the HRD-profiler criterion can "
                    "expose how the L2 prefetch hint changes "
                    "effective latency. The PTX intrinsic is "
                    "supported on CC >= 8.0 and admits on the "
                    "sm_120 baseline (no `sm_120a` / `sm_120f` "
                    "conditional). R53 captured canonical live "
                    "sm_120 trace + per-shape sidecars on the "
                    "RTX 5060 Laptop GPU via `run_suite.py "
                    "--mode trace`: workloads/simbench_bw/results/"
                    "l2_prefetch_hint_sweep/blackwell_consumer_unknown/"
                    "sm_120/ admits via verify_artifacts.py "
                    "(trace_trace_00.json + manifest + dynamic "
                    "trace + the per-shape feature sidecars (sector_stream, hrd_profile, divergence_proxies, sharing_locality, order_projections, perturbation_run_ledger, convergence_confidence)). "
                    "The artifact tree rebases on every commit "
                    "per the abstract-HEAD pattern."
                    "Live sm_86 cross-arch evidence captured on the RTX "
                    "3090 Ti (Ampere desktop, GA102, CC 8.6, NVBit 1.8, "
                    "CUDA 13.2, native g++-14) under "
                    "workloads/simbench_bw/results/l2_prefetch_hint_sweep/rtx_3090_ti/sm_86/ "
                    "via the same `run_suite.py --mode trace` path "
                    "post-arch_gate widening (the slug's `arch_gate` was "
                    "widened to admit sm_86 in `roster.toml`, `Makefile`, "
                    "and the `.cu` source comment); admits via "
                    "`verify_artifacts.py` and carries the per-shape "
                    "feature sidecars (sector_stream, hrd_profile, "
                    "divergence_proxies, sharing_locality, "
                    "order_projections); the typed-fail-closed arms "
                    "(e.g., `hrd_layer_unavailable` when the upstream HRD "
                    "lacks usable admitted bins for one or both layers) "
                    "fire when triggered per the per-arch L1/L2 sizing "
                    "differences. "
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="l2_persisting_window_sweep",
        plan_description=(
            "L2 persisting-window sweep where supported."
        ),
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="l2_persisting_window_sweep",
                roster="simbench",
                rationale=(
                    "R47 landing + R48 harness-visible typed-"
                    "unavailable wiring: the host driver sets an "
                    "L2 persistence access-policy window via "
                    "`cudaStreamSetAttribute` + "
                    "`cudaStreamAttributeAccessPolicyWindow` over "
                    "the input allocation before launch. The "
                    "shape file's `persisting_window_bytes` axis "
                    "sweeps the requested window size; the runtime "
                    "clips to the device's actual L2 persisting "
                    "capacity (queryable via "
                    "`cudaDevAttrMaxAccessPolicyWindowSize`). On "
                    "architectures that do not advertise the "
                    "attribute the host driver emits a structured "
                    "typed-unavailable JSON document on stdout "
                    "(`result_kind == \"typed_unavailable\"`, "
                    "`typed_reason == \"arch_feature_unavailable_l2_persisting\"`); "
                    "`run_suite.py`'s correctness path detects the "
                    "result_kind and persists the typed-unavailable "
                    "candidate JSON without comparing to the "
                    "reference. R53 captured canonical live "
                    "sm_120 trace + per-shape sidecars on the RTX "
                    "5060 Laptop GPU via `run_suite.py --mode "
                    "trace`: workloads/simbench_bw/results/"
                    "l2_persisting_window_sweep/blackwell_consumer_unknown/"
                    "sm_120/ admits via verify_artifacts.py "
                    "(trace_trace_00.json + manifest + dynamic "
                    "trace + the per-shape feature sidecars (sector_stream, hrd_profile, divergence_proxies, sharing_locality, order_projections, perturbation_run_ledger, convergence_confidence)). "
                    "The device advertised the persistence "
                    "attribute and the kernel ran natively, not "
                    "through the typed-unavailable path. The "
                    "artifact tree rebases on every commit per "
                    "the abstract-HEAD pattern."
                    "Live sm_86 cross-arch evidence captured on the RTX "
                    "3090 Ti (Ampere desktop, GA102, CC 8.6, NVBit 1.8, "
                    "CUDA 13.2, native g++-14) under "
                    "workloads/simbench_bw/results/l2_persisting_window_sweep/rtx_3090_ti/sm_86/ "
                    "via the same `run_suite.py --mode trace` path "
                    "post-arch_gate widening (the slug's `arch_gate` was "
                    "widened to admit sm_86 in `roster.toml`, `Makefile`, "
                    "and the `.cu` source comment); admits via "
                    "`verify_artifacts.py` and carries the per-shape "
                    "feature sidecars (sector_stream, hrd_profile, "
                    "divergence_proxies, sharing_locality, "
                    "order_projections); the typed-fail-closed arms "
                    "(e.g., `hrd_layer_unavailable` when the upstream HRD "
                    "lacks usable admitted bins for one or both layers) "
                    "fire when triggered per the per-arch L1/L2 sizing "
                    "differences. "
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="carveout_sensitivity_probe",
        plan_description="Carveout sensitivity probe.",
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="carveout_sensitivity_probe",
                roster="simbench",
                rationale=(
                    "R48 landing + R49 multi-shape sweep + L1/"
                    "shared-pressure kernel rework: the host "
                    "driver invokes `cudaFuncSetAttribute("
                    "cudaFuncAttributePreferredSharedMemoryCarveout, "
                    "carveout_percent)` before launch followed by "
                    "`cudaFuncSetAttribute("
                    "cudaFuncAttributeMaxDynamicSharedMemorySize, "
                    "shared_bytes)` to request the carveout-"
                    "bounded shared budget. The shape file's "
                    "`carveout_percent` axis now sweeps three "
                    "trace shapes (25 / 50 / 75 percent) plus "
                    "matching timing shapes; the kernel reads "
                    "from BOTH the multi-KiB carveout-bounded "
                    "shared tile and the global allocation so "
                    "the partition split is observable in L1 "
                    "traffic. The preference is hardware-"
                    "advisory and the runtime clips to the "
                    "device's supported partition points "
                    "(typically 0/25/50/75/100 percent). On "
                    "architectures that do not advertise per-"
                    "kernel carveout selection (queried via "
                    "`cudaDevAttrMaxSharedMemoryPerBlockOptin`) "
                    "the host driver emits a structured typed-"
                    "unavailable JSON document on stdout "
                    "(`result_kind == \"typed_unavailable\"`, "
                    "`typed_reason == \"arch_feature_unavailable_carveout\"`); "
                    "`run_suite.py`'s correctness path detects the "
                    "result_kind and persists the typed-unavailable "
                    "candidate JSON without comparing to the "
                    "reference (R49 extends the same wiring to "
                    "trace and timing modes). R52 captured "
                    "canonical live sm_120 trace + per-shape "
                    "sidecars on the RTX 5060 Laptop GPU via "
                    "`run_suite.py --mode trace` for ALL THREE "
                    "carveout shape points (trace_00, trace_50, "
                    "trace_75): workloads/simbench_bw/results/"
                    "carveout_sensitivity_probe/blackwell_consumer_unknown/"
                    "sm_120/ admits via verify_artifacts.py for "
                    "all three shapes (trace_trace_{00,50,75}"
                    ".json + manifest + per-shape "
                    "trace_trace_{00,50,75}/dynamic_trace.pb + "
                    "the per-shape feature sidecars). The "
                    "device advertised the carveout attribute "
                    "and the kernel ran natively, not through "
                    "the typed-unavailable path. The artifact "
                    "tree rebases on every commit per the "
                    "abstract-HEAD pattern. After R52 every "
                    "microbench-suite-coverage family except "
                    "`warm_cold_pair` has canonical "
                    "`live_verified_sm_120` evidence; "
                    "warm_cold_pair retains its R50 raw "
                    "evidence pending an HRD-producer "
                    "investigation (the multi-launch trace "
                    "expansion appears O(n²) in the "
                    "reuse-distance computation)."
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="bandwidth_saturation_sweep",
        plan_description="Bandwidth saturation sweep.",
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="dram_stream_stride_mix",
                roster="simbench",
                rationale=(
                    "The slug's `timing_shapes` axes "
                    "(`warps=1024, accesses=4096, "
                    "stride_classes=4`) are sized to drive DRAM "
                    "saturation; sweeping the `accesses` axis "
                    "across orders of magnitude exercises the knee "
                    "from compute-bound to memory-bound. The "
                    "divergence-proxy criterion emits "
                    "`md_loads_per_kilo_inst` and `stride_entropy` "
                    "for this slug, which together pin the "
                    "saturation regime. Live sm_120 evidence "
                    "rebases on every commit per the abstract-HEAD "
                    "pattern."
                ),
            ),
        ),
    ),
    Ac14Family(
        family_id="resident_warp_pressure_sweep",
        plan_description="Resident-warp pressure sweep.",
        state=Ac14FamilyState.live_verified_sm_120,
        covered_slugs=(
            Ac14SlugCoverage(
                slug="register_spill_local_mem",
                roster="simbench",
                rationale=(
                    "Per-thread register pressure drives spill to "
                    "local memory, which in turn changes resident-"
                    "warp count under the SM occupancy budget. "
                    "Sweeping the `block` dimension in the shape "
                    "file's `launch_config` exercises the resident-"
                    "warp envelope; the divergence-proxy "
                    "criterion's `outstanding_load_density_by_phase` "
                    "reports the pressure signal. R52 captured "
                    "canonical live sm_120 trace + per-shape "
                    "sidecars on the RTX 5060 Laptop GPU via "
                    "`run_suite.py --mode trace`: "
                    "workloads/simbench_bw/results/"
                    "register_spill_local_mem/blackwell_consumer_unknown/"
                    "sm_120/ admits via verify_artifacts.py "
                    "(trace_trace_00.json + manifest + dynamic "
                    "trace + the per-shape feature sidecars (sector_stream, hrd_profile, divergence_proxies, sharing_locality, order_projections, perturbation_run_ledger, convergence_confidence)). "
                    "The artifact tree rebases on every commit "
                    "per the abstract-HEAD pattern."
                    "Live sm_86 cross-arch evidence captured on the RTX "
                    "3090 Ti (Ampere desktop, GA102, CC 8.6, NVBit 1.8, "
                    "CUDA 13.2, native g++-14) under "
                    "workloads/simbench_bw/results/register_spill_local_mem/rtx_3090_ti/sm_86/ "
                    "via the same `run_suite.py --mode trace` path "
                    "post-arch_gate widening (the slug's `arch_gate` was "
                    "widened to admit sm_86 in `roster.toml`, `Makefile`, "
                    "and the `.cu` source comment); admits via "
                    "`verify_artifacts.py` and carries the per-shape "
                    "feature sidecars (sector_stream, hrd_profile, "
                    "divergence_proxies, sharing_locality, "
                    "order_projections); the typed-fail-closed arms "
                    "(e.g., `hrd_layer_unavailable` when the upstream HRD "
                    "lacks usable admitted bins for one or both layers) "
                    "fire when triggered per the per-arch L1/L2 sizing "
                    "differences. "
                ),
            ),
        ),
    ),
)


def families() -> tuple[Ac14Family, ...]:
    """Return the canonical family tuple. The validator's exact-
    count assertion uses this to lock the plan-enumerated set at
    exactly 10 entries (the cache-modifier `.ca`/`.cg`/`.cs` plan
    row is recorded as one inventory family `cache_modifier_variants`;
    the per-variant SASS evidence surface lives in the
    perturbation-harness clause's typed-reason set)."""
    return _AC14_FAMILIES


def family_count() -> int:
    """The plan-enumerated family count is exactly 10. The
    cache-modifier `.ca`/`.cg`/`.cs` plan row is recorded as a
    single family `cache_modifier_variants`; the per-variant SASS
    evidence surface is exposed by the perturbation-harness clause
    rather than as separate inventory rows."""
    return 10


def families_in_state(state: Ac14FamilyState) -> tuple[Ac14Family, ...]:
    """Return the families currently in the given state."""
    return tuple(f for f in _AC14_FAMILIES if f.state == state)


def covered_slugs(*, family_id: str | None = None) -> tuple[str, ...]:
    """Return the deduplicated set of slugs that cover the given
    family (or all families when `family_id` is None). Only
    families in `live_verified_sm_120` or `structural_only`
    states contribute slugs; `arch_unsupported` and
    `not_implemented` families do not."""
    out: list[str] = []
    eligible = (
        Ac14FamilyState.live_verified_sm_120,
        Ac14FamilyState.structural_only,
    )
    for family in _AC14_FAMILIES:
        if family.state not in eligible:
            continue
        if family_id is not None and family.family_id != family_id:
            continue
        for cov in family.covered_slugs:
            if cov.slug not in out:
                out.append(cov.slug)
    return tuple(out)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def simbench_roster_path() -> Path:
    """The simbench roster file the validator inspects."""
    return _REPO_ROOT / "workloads" / "simbench_bw" / "roster.toml"


def portable_lcd_roster_path() -> Path:
    """The portable_lcd roster file the validator inspects."""
    return _REPO_ROOT / "workloads" / "portable_lcd" / "roster.toml"


def simbench_shape_file(slug: str) -> Path:
    """Path of the simbench shape file the validator inspects."""
    return (
        _REPO_ROOT
        / "workloads" / "simbench_bw" / "shapes"
        / f"{slug}.toml"
    )


REQUIRED_SHAPE_BLOCKS: tuple[str, ...] = (
    "timing_shapes",
    "trace_shapes",
    "correctness_shapes",
)
