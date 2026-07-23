"""Thin harness wrapping the official ``sol-execbench`` evaluator CLI.

The harness does NOT reimplement evaluation. It shells out to the
native CLI as ``sol-execbench <problem_dir> --solution <solution.json>
--json``, parses the JSONL ``Trace`` lines the upstream CLI emits, and
produces a per-workload report row per landed baseline. Reward-hacking
detection, GPU clock policy, and timing methodology all live upstream
— the harness merely wraps, parses, and routes results.

Deferred-ledger schema (``deferred.json``) — one JSON file per
baseline tree (or a single top-level ledger) with rows like:

.. code-block:: json

    {
      "slug": "problem_001_fp8_attention_output_projection",
      "language": "cuda_cpp",
      "status": "deferred_correctness",
      "justification": "INCORRECT_NUMERICAL on every workload (...)",
      "evidence": ".humanize/rlcr/.../round-N-trace.jsonl#L42",
      "harness_commit": "<sha>"
    }

Under the correctness-only landing contract, valid statuses are:
``deferred_correctness`` (any non-PASSED status other than
REWARD_HACK) and ``deferred_reward_hack`` (status == REWARD_HACK).
Slow-but-correct kernels land — there is no ``deferred_slow``
status. Pilots in the deferred ledger are NOT committed under
``baselines/``; they remain in a staging area or in the deferred
ledger only.
"""
