"""Counter-free analytics framework metrics + cache-prediction
contract surfaces.

This package houses the metrics-side modules the deployed
analytics framework relies on, organised by area of concern:

* **Cache predictor (deployed per-launch predictor body)** —
  ``cache_predictor.py`` ships the deployed L1 / L2 hit-rate
  predictor. ``predict_from_launch_frame`` consumes a parsed
  Tier-2 sidecar ``LaunchFrame``, routes each compact sector
  record through ``cache_access_policy.route_access`` to decide
  L1 / L2 participation, replays the eligible 32 B sector ids
  through two ``SetAssociativeSimulator`` instances (one for L1,
  one for L2), and returns raw per-layer hit-rates + hit / total
  counts + provenance. ``predict_warmed_launch_sequence`` walks
  an ordered launch sequence reusing the same simulator state
  across launches so the NCU ``--launch-skip N --launch-count 1``
  warm-replay protocol can be reproduced on the deployed path.
  ``compose_predictor_bundle`` turns either result dict into a
  ``PredictorBundle`` (scalar arm or typed-unavailable arm)
  suitable for ``predictor_bundle_schema.validate_predictor_bundle``.
  The four canonical typed-unavailable fail-closed branches are
  ``trace_budget_exceeded`` (truncated / sampled trace),
  ``policy_uncalibrated`` (unknown routing),
  ``no_admissible_accesses`` (layer with zero admissible
  accesses), and ``cache_geometry_unresolved``. The module
  imports only stdlib + sibling ``baselines_harness.metrics``
  modules; no NCU dependency at runtime.
* **Cache geometry and access policy** — ``cache_geometry.py``
  and ``cache_access_policy.py`` carry the per-arch geometry
  descriptors and the L1 / L2 access-policy enums the simulator
  + predictor surfaces depend on.
* **Hardware-descriptor input boundary** —
  ``hardware_descriptor.py`` exposes the frozen
  ``HardwareDescriptor`` dataclass + ``from_cache_geometry``
  adapter; the canonical input contract for the cross-arch
  pooled head.
* **Pooled cross-arch head** — ``pooled_cross_arch_head.py``
  owns the validated training-boundary stub
  (``fit_pooled_head``), the descriptor → numeric design matrix
  builder (``build_pooled_head_feature_matrix``), the
  deployment-provenance contract dataclass
  (``PooledHeadFitResult`` with per-field + cross-field
  validators), the canonical B200 zero-shot factory
  (``make_b200_zero_shot_fit_result``), and the closed
  vocabulary frozensets for calibration sources, prediction
  modes, and typed reasons. ``fit_pooled_head`` first
  validates its descriptor sequence via
  ``build_pooled_head_feature_matrix`` and its label sequence
  via ``_require_valid_label_sequence``, then raises
  ``PooledHeadTrainingNotAvailableError`` with typed reason
  ``pooled_head_training_labels_not_yet_harvested`` until
  sm_120 NCU calibration labels are available on real
  hardware. ``fit_pooled_head`` is the only remaining stub in
  this package: the input-validation surface is fully
  implemented; only the actual fitting algorithm is
  hardware-gated. The deployed per-launch cache predictor in
  ``cache_predictor.py`` ships independently of this gate (see
  the cache-predictor bullet above).
* **Framework / bundle / sidecar contracts** —
  ``framework_contract.py`` (cell join key + cohort
  identifiers), ``predictor_bundle_schema.py`` (
  ``PredictorBundle`` schema + ``validate_predictor_bundle``),
  ``predictor_design_coverage.py`` (predictor-friendly /
  predictor-hostile cohort tags), and ``tier2_sidecar_parser.py``
  (NVBit Tier-2 trace sidecar parser).
* **Tiered-gate evaluator + saturation regression guard** —
  ``tiered_gate_evaluator.py`` (cohort-aware
  ``±5 / ±15 pp`` tiered accuracy gate) and
  ``saturation_regression_guard.py``.
* **Calibration head ensemble + cohort classifier** —
  ``isotonic_calibrator.py`` (``IsotonicCalibrator`` shape),
  ``sharing_classifier.py`` (intra-warp / intra-CTA /
  inter-CTA / cold bucket classification),
  ``upper_bound_calibrator.py`` (Maeda 2017 multi-granularity
  HRD overlay-adjacent calibrator),
  ``hrd_multigranularity_overlay.py`` (sector / line / page
  reuse-distance overlay).
* **Set-associative simulator** —
  ``set_associative_simulator.py`` carries the Zhang USENIX'24
  two-phase replacement simulator.
* **Cold-cache warm-up protocol** —
  ``warmup_protocol.py`` pins the cold / warm cache-initial-
  state vocabulary the calibration corpus is captured under.
* **Typed-unavailable bundle composition** —
  ``typed_unavailable_bundle.py`` composes typed-fail-close
  bundles for cells the predictor cannot honestly score.

The package's ``__all__`` is intentionally narrow at
``["cache_predictor"]`` — the deployed per-launch predictor
module is the only symbol exposed for the wildcard
``from baselines_harness.metrics import *`` import. Every
other module must be imported explicitly via
``from baselines_harness.metrics.<module> import <name>``;
this keeps the wildcard surface from accidentally exposing
internal helpers.

Per CLAUDE.md §"Critical Rules" Rule 1, this package's deployed
runtime path is NCU-free; ``cache_predictor.py`` and every
other deployed module must keep the NCU-free invariant.
Calibration-only NCU access lives under
``tools/sol_baselines_harness/calibration/cache_prediction/`` and
is policed by ``tests/test_ncu_quarantine.py``.
"""

__all__ = ["cache_predictor"]
