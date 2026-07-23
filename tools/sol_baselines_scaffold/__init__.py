"""Scaffold + pack + validate toolchain for SOL-ExecBench baseline kernels.

This package supersedes the prior ``baselines_codegen`` module (renamed via
``git mv`` so history is preserved). It exposes three independent concerns
plus a preserved manifest/metadata builder:

- ``scaffold.py`` — copy an exemplar tree from
  ``sol-execbench/examples/{cutlass,cuda_cpp}/`` into
  ``baselines/<category>/problem_NNN_<slug>/<language>/``, convert the entry
  function to destination-passing style, rename kernel symbols, and emit a
  Pydantic-valid ``solution.json``.
- ``pack.py`` — re-pack ``solution.json.sources[].content`` from the editable
  files on disk. Disk files are the canonical source of truth; embedded JSON
  content is derived. Deterministic source order (entry first, remaining
  lexicographically by ``path``) keeps re-runs byte-identical.
- ``validate.py`` + ``test_contract.py`` — parse every emitted
  ``solution.json`` against the live ``Solution`` / ``Definition`` Pydantic
  models (schema layer) and apply project lint that the live schema permits
  (Rule-5 vendor-language rejection, Rule-1/2/6/7 forbidden-surface scan,
  destination-passing-style argument-order check against ``definition.json``).
- ``generate.py`` — manifest mode (``--manifest-only``) and the
  slug-resolution + subset-mapping primitives consumed by ``scaffold.py``.
  The legacy ``--slug`` + ``--language`` template-driven codegen path
  was removed in the same commit that landed pilot #1 under
  ``baselines/FIB/problem_001_fused_add_rmsnorm_h2048/``; per-baseline
  emission is now ``scaffold.py``'s responsibility.

The slug-resolution heuristic strips ``^problem_`` (not ``^problem_\\d+_``):
internal slug ``problem_001_fused_add_rmsnorm_h2048`` resolves to the
downloader's directory name ``001_fused_add_rmsnorm_h2048`` under
``sol-execbench/data/benchmark/FlashInfer-Bench/``.
"""
