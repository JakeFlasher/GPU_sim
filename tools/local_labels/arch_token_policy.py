"""Single source of truth for the arch-token boundary mapping.

The Cross-Arch Bounded First sm_120 Label Admission Loop pins three
distinct arch tokens that live at different layers of the system; the
boundary mapping is deterministic and must be honored consistently by
every producer + consumer that crosses a layer.

The three tokens:

1. **Label / comparison token** (`sm_86`, `sm_120`).
   Used in ``data/local_labels/label_admitted.json``'s ``arch`` field;
   in ``workloads/simbench_bw/comparison_contract.toml``'s ``arch_set``;
   in ``workloads/simbench_bw/comparison_runner.py``'s 7-field join key.
   The v4 admitted-label schema's ``arch`` enum admits ONLY these two
   tokens.

2. **Device / capability / sidecar-directory token** (`sm_86`,
   `sm_120-laptop`). Used in
   ``tools/arch_capabilities.json``'s ``supported_arches`` list, in
   ``tools/hrd_profiler/device_properties.json`` keys, and in
   ``data/local_labels/launch_parameters_sidecars/<arch>/``
   subdirectory names. ``label_driver.sh:213-237`` already maps
   ``sm_120 -> sm_120-laptop`` at the capability gate as the
   precedent.

3. **Trace / results platform-path token** (e.g.,
   ``rtx_3090_ti/sm_86``, ``rtx_5060_laptop/sm_120``). Used in result /
   trace path templates that follow the existing
   ``results/<slug>/<sku>/<arch>/`` shape (the ``<sku>`` segment is the
   SKU registered in
   ``workloads/simbench_bw/arch_resolver.py::PLATFORM_TO_CC_FAMILY``).

The two maps below encode the boundary mapping. New label arches added
in the future MUST extend BOTH maps in lockstep (and the registry
surfaces enumerated in
``workloads/simbench_bw/tests/test_rtx_5060_laptop_platform_registry_lockstep.py``)
to avoid the silent admission gap that prompted this module.
"""

from __future__ import annotations

# Label arch -> sidecar-directory token. Used by:
#   - tools/local_labels/extend_launch_parameters.py to route output to
#     data/local_labels/launch_parameters_sidecars/<sidecar_token>/
#   - tools/local_labels/run_headline_cohort_predict_compare.py to
#     locate the sidecar by arch at lookup time
LABEL_ARCH_TO_SIDECAR_TOKEN: dict[str, str] = {
    "sm_86": "sm_86",
    "sm_120": "sm_120-laptop",
}


# Label arch -> trace/results platform path (sku/arch). Used by:
#   - tools/local_labels/extend_launch_parameters.py to locate the
#     trace directory at workloads/simbench_bw/results/<slug>/<path>/
LABEL_ARCH_TO_PLATFORM_PATH: dict[str, str] = {
    "sm_86": "rtx_3090_ti/sm_86",
    "sm_120": "rtx_5060_laptop/sm_120",
}


# Arches that admit through the label / comparison surface. Mirrors the
# v4 admitted-label schema's arch enum + comparison_contract.toml's
# arch_set; the source of truth for that enum lives in the schema.
SUPPORTED_LABEL_ARCHES: frozenset[str] = frozenset(LABEL_ARCH_TO_SIDECAR_TOKEN)


class UnsupportedLabelArchError(ValueError):
    """Raised when a label arch is encountered that is not in
    :data:`SUPPORTED_LABEL_ARCHES`. Distinct from
    :class:`KeyError` so callers can fail-closed without ambiguity."""


class SidecarDirectoryMissingError(FileNotFoundError):
    """Raised by a predictor-side launch-parameter lookup when the
    arch's sidecar directory is not present on disk. Distinct from a
    missing-individual-sidecar fall-through: sm_86 may legitimately
    have cells without committed sidecars (legacy fall-through admits
    a typed no-correction path); sm_120 cannot silently skip gridDim
    correction because that would be a silent miscalibration. The
    typed exception forces the caller to either populate the directory
    or document an explicit exclusion."""


def sidecar_token_for_label_arch(arch: str) -> str:
    """Return the sidecar-directory token for the given label arch.
    Raises :class:`UnsupportedLabelArchError` on unknown arch."""
    try:
        return LABEL_ARCH_TO_SIDECAR_TOKEN[arch]
    except KeyError as exc:
        raise UnsupportedLabelArchError(
            f"Unknown label arch {arch!r}; supported arches: "
            f"{sorted(SUPPORTED_LABEL_ARCHES)}. The arch must match the "
            f"v4 admitted-label schema's arch enum."
        ) from exc


def platform_path_for_label_arch(arch: str) -> str:
    """Return the trace/results platform-path segment for the given
    label arch. Raises :class:`UnsupportedLabelArchError` on unknown
    arch."""
    try:
        return LABEL_ARCH_TO_PLATFORM_PATH[arch]
    except KeyError as exc:
        raise UnsupportedLabelArchError(
            f"Unknown label arch {arch!r}; supported arches: "
            f"{sorted(SUPPORTED_LABEL_ARCHES)}. The arch must match the "
            f"v4 admitted-label schema's arch enum."
        ) from exc
