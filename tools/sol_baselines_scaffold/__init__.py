"""Scaffold + pack + validate toolchain for SOL-ExecBench baseline kernels.

It exposes the scaffold, pack, validate, and manifest/metadata builders:

- ``scaffold.py`` — copy an exemplar tree from
  ``sol-execbench/examples/{cutlass,cuda_cpp}/`` into
  ``baselines/<category>/problem_NNN_<slug>/<language>/``, convert the entry
  function to destination-passing style, rename kernel symbols, and emit a
  Pydantic-valid ``solution.json``.
- ``pack.py`` — re-pack ``solution.json.sources[].content`` from the editable
  files on disk. Disk files are the canonical source of truth; embedded JSON
  content is derived. Deterministic source order (entry first, remaining
  lexicographically by ``path``) keeps re-runs byte-identical.
- ``validate.py`` + focused contract tests — parse every emitted
  ``solution.json`` against the live ``Solution`` / ``Definition`` Pydantic
  models (schema layer) and apply project lint that the live schema permits
  (Rule-5 vendor-language rejection, Rule-1/2/6/7 forbidden-surface scan,
  destination-passing-style argument-order check against ``definition.json``).
- ``generate.py`` — manifest mode (``--manifest-only``) and the
  slug-resolution + subset-mapping primitives consumed by ``scaffold.py``.

The slug-resolution heuristic strips ``^problem_`` (not ``^problem_\\d+_``):
internal slug ``problem_001_fused_add_rmsnorm_h2048`` resolves to the
downloader's directory name ``001_fused_add_rmsnorm_h2048`` under
``sol-execbench/data/benchmark/FlashInfer-Bench/``.
"""

import sys as _sys

_sys.modules.setdefault("baselines_scaffold", _sys.modules[__name__])
