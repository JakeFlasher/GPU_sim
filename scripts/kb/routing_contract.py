"""Shared knowledge-router ownership vocabulary.

This module contains only stable data so runtime routing and SOL authoring tools
can import the same contract without importing either command-line program.
"""

from __future__ import annotations

SUBCORPUS_TO_SKILL = {
    "ptx": "interpreting-ptx-isa",
    "inline-ptx": "interpreting-ptx-isa",
    "sass": "analyzing-sass-nvbit-traces",
    "cuda": "optimizing-cuda-kernels",
    "best-practices": "optimizing-cuda-kernels",
    "blackwell": "tuning-blackwell-sm120-kernels",
    # Active technical skill for profiler permissions and measurement choices.
    "measurement": "validating-nonroot-gpu-measurement",
    "patents": "reasoning-about-nvidia-architecture-history",
    "simulator": "designing-gpu-simulator",
    "nvcc": "compiling-cuda-artifacts",
    "sources": "implementing-sol-baselines",
    "patterns": "implementing-sol-baselines",
    "problems": "implementing-sol-baselines",
    "cutlass_constructs": "implementing-sol-baselines",
    # Root-level curation such as INDEX.md is owned by the meta-router.
    "_root": "navigating-project-kb",
}

CANONICAL_SKILLS = frozenset(SUBCORPUS_TO_SKILL.values())
CANONICAL_SUBCORPORA = frozenset(SUBCORPUS_TO_SKILL)
