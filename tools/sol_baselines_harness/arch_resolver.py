"""Resolve the live host's device identity to a SOLAR arch yaml.

Before this module landed, ``score.py`` and ``solar_analysis.py``
hardcoded ``SOLAR/configs/arch/5090.yaml`` regardless of the live
device. On a 3090 Ti (sm_86 / GA102) that produced numerically
plausible but architecturally wrong ``local_sol_score`` values
against a non-matching roofline.

The resolver first distinguishes CC 12.0 device tiers. The RTX 5090
Workstation Edition is the active cloud target, but it has no checked-in SOLAR
roofline yet, so it intentionally returns ``None`` rather than selecting the
historical RTX 5090 yaml. Unknown / malformed / MIG inputs return
``None`` from ``resolve_local_arch_yaml`` and ``"sm_unknown"`` (with
a diagnostic when applicable) from ``arch_tag_from_host_kind``; the
caller is responsible for short-circuiting numeric scoring to
``unknown_t_sol`` in that case.

The mapping is intentionally explicit. Adding a new arch is a
one-line edit here plus authoring the corresponding yaml under
``SOLAR/configs/arch/``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Mapping, Optional

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_ARCH_ROOT = _REPO_ROOT / "tools" / "solar" / "configs" / "arch"
if not _ARCH_ROOT.is_dir():
    _ARCH_ROOT = _REPO_ROOT / "SOLAR" / "configs" / "arch"

_LOGGER = logging.getLogger(__name__)


# Compute-capability major.minor → arch yaml filename, relative to
# ``SOLAR/configs/arch/``. The mapping is closed: anything not
# enumerated here returns ``None``, which the caller renders as
# ``unknown_t_sol``.
_CC_TO_YAML: Mapping[str, str] = {
    "8.6": "3090ti.yaml",
    "9.0": "H100_PCIe.yaml",
    "10.0": "B200.yaml",
    "10.0a": "B200.yaml",
}


_CC_PATTERN = re.compile(r"^(\d+)\.(\d+)([a-z]?)$")


def _normalise_cc(compute_capability: Optional[str]) -> Optional[str]:
    """Strip whitespace and accept ``X.Y`` / ``X.Ya`` forms; reject
    everything else.

    Returns the canonical form (e.g. ``"8.6"`` or ``"10.0a"``) when
    the input matches; otherwise ``None``.
    """
    if compute_capability is None:
        return None
    raw = compute_capability.strip()
    if not raw:
        return None
    if raw.lower() == "unknown":
        return None
    match = _CC_PATTERN.match(raw)
    if not match:
        return None
    major, minor, suffix = match.group(1), match.group(2), match.group(3)
    return f"{major}.{minor}{suffix}"


def _is_mig_or_partitioned(host_info: object) -> bool:
    """Heuristic MIG / partitioned-device detection.

    The orchestrator's ``HostInfo`` is a dataclass; this module
    accepts it via duck typing so the resolver can also be called
    with a plain dict (e.g. in tests). Returns ``True`` when the
    device_name field carries the standard NVIDIA MIG marker
    (``"MIG "`` prefix or ``"-mig-"`` substring), which indicates
    the visible device is a slice — sustained throughput on a slice
    requires per-slice arch math that this repo does not currently
    model.
    """
    device_name = ""
    if isinstance(host_info, dict):
        device_name = str(host_info.get("device_name", ""))
    else:
        device_name = str(getattr(host_info, "device_name", ""))
    if not device_name:
        return False
    lowered = device_name.lower()
    if device_name.startswith("MIG "):
        return True
    if "-mig-" in lowered:
        return True
    return False


def resolve_local_arch_yaml(host_info: object) -> Optional[Path]:
    """Return the SOLAR arch yaml path for the local device.

    ``host_info`` is the orchestrator's ``HostInfo`` (duck-typed; a
    dict with ``compute_capability`` and ``device_name`` keys also
    works). Returns ``None`` and logs a diagnostic when:

    - the compute capability is missing / malformed / ``"unknown"``;
    - the compute capability is not in the closed mapping;
    - the device is a MIG / partitioned slice (per
      ``_is_mig_or_partitioned``).

    Callers must treat ``None`` as ``unknown_t_sol`` for the
    scoring path.
    """
    cc_raw = None
    device_name = ""
    if isinstance(host_info, dict):
        cc_raw = host_info.get("compute_capability")
        device_name = str(host_info.get("device_name", ""))
    else:
        cc_raw = getattr(host_info, "compute_capability", None)
        device_name = str(getattr(host_info, "device_name", ""))

    if _is_mig_or_partitioned(host_info):
        _LOGGER.warning(
            "MIG/partitioned device not supported in arch resolver; "
            "SOL Score will be unknown_t_sol"
        )
        return None

    cc = _normalise_cc(cc_raw if isinstance(cc_raw, str) else None)
    if cc is None:
        return None
    if cc == "12.0":
        lowered = device_name.lower()
        if "rtx 5090" in lowered:
            _LOGGER.info(
                "SM120 RTX 5090 detected; no RTX-5090-specific SOLAR yaml is installed"
            )
            return None
        if "5090" in lowered:
            legacy_yaml = _ARCH_ROOT / "5090.yaml"
            return legacy_yaml if legacy_yaml.is_file() else None
        _LOGGER.warning(
            "CC 12.0 device %r is not a known card tier; SOL Score is unknown_t_sol",
            device_name,
        )
        return None
    yaml_name = _CC_TO_YAML.get(cc)
    if yaml_name is None:
        return None
    yaml_path = _ARCH_ROOT / yaml_name
    if not yaml_path.is_file():
        _LOGGER.warning(
            "arch yaml mapped to %s but file is missing on disk; "
            "SOL Score will be unknown_t_sol",
            yaml_path,
        )
        return None
    return yaml_path


# Host-kind → arch tag mapping. The host-kind vocabulary is defined
# in ``surfaces/host_probe.py:_classify_host_kind`` and emits values
# like ``"local_sm_8_6"``, ``"local_sm_120"``, ``"b200_sm_100a"``.
# The arch_tag is what appears in the manifest evidence label
# ("Compile-checked-sm_86", "Profiled-B200-sm_100a", etc.) and in
# the report's ``arch_tag`` column.
_KIND_KNOWN_OVERRIDES: Mapping[str, str] = {
    "b200_sm_100a": "sm_100a",
    "local_sm_120": "sm_120",
    "local_sm_90": "sm_90",
}


_KIND_LOCAL_SM_PATTERN = re.compile(
    r"^local_sm_(\d+)(?:_(\d+))?$"
)


def arch_tag_from_host_kind(kind: Optional[str]) -> str:
    """Translate the orchestrator's host kind to a manifest arch tag.

    Examples:
        ``"local_sm_8_6"``    → ``"sm_86"``
        ``"local_sm_9_0"``    → ``"sm_90"``
        ``"local_sm_120"``    → ``"sm_120"``
        ``"b200_sm_100a"``    → ``"sm_100a"``
        ``"unknown"``         → ``"sm_unknown"``

    Anything else returns ``"sm_unknown"``.
    """
    if not isinstance(kind, str) or not kind:
        return "sm_unknown"
    if kind in _KIND_KNOWN_OVERRIDES:
        return _KIND_KNOWN_OVERRIDES[kind]
    match = _KIND_LOCAL_SM_PATTERN.match(kind)
    if match is not None:
        major, minor = match.group(1), match.group(2)
        if minor is None:
            # `local_sm_120` shape (digits glued).
            return f"sm_{major}"
        return f"sm_{major}{minor}"
    return "sm_unknown"


def arch_resolution_reason(host_info: object) -> str:
    """Return a short human-readable string describing the resolution.

    Used by ``report.py`` to populate the per-row
    ``arch_resolution_reason`` column so downstream consumers can
    trace which yaml a numeric score was computed against.
    """
    cc_raw = None
    device_name = ""
    if isinstance(host_info, dict):
        cc_raw = host_info.get("compute_capability")
        device_name = str(host_info.get("device_name", ""))
    else:
        cc_raw = getattr(host_info, "compute_capability", None)
        device_name = str(getattr(host_info, "device_name", ""))
    if _is_mig_or_partitioned(host_info):
        return "MIG/partitioned device; arch yaml unsupported"
    cc = _normalise_cc(cc_raw if isinstance(cc_raw, str) else None)
    if cc is None:
        return f"compute_capability={cc_raw!r} unrecognised"
    if cc == "12.0":
        lowered = device_name.lower()
        if "rtx 5090" in lowered:
            return "compute_capability=12.0; RTX 5090 has no SOLAR yaml"
        if "5090" in lowered:
            return "compute_capability=12.0; historical RTX 5090 yaml"
        return f"compute_capability=12.0; unknown device tier {device_name!r}"
    if cc not in _CC_TO_YAML:
        return f"compute_capability={cc} unrecognised"
    return f"compute_capability={cc}"
