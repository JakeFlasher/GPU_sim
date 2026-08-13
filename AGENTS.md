# Research-first agent policy

This repository is an experimental GPU-simulator project. Agents must advance
the requested research or implementation directly and keep process overhead
proportional to the work.

## Required working style

- Inspect the relevant code and data, make the requested change, and run only
  the ordinary checks needed to catch implementation mistakes.
- NVIDIA NCU, NVBit, CUDA tools, Rust/Python toolchains, GPU execution, and
  operator-authorized `sudo` commands are available project tools. Do not
  disable or quarantine them merely because they execute native code or access
  hardware.
- Use focused unit, integration, build, or experiment checks with bounded
  runtimes. A failed check is a local engineering problem to diagnose and fix.
- Ask the user only when a materially different research outcome requires a
  user-owned choice or a necessary external fact is unavailable locally.
- Do not spawn review or audit subagents unless the user explicitly requests
  parallel agent work.

## Permanently prohibited workflow machinery

Do not create, restore, or enforce any of the following:

- audit, adjudication, verifier, or review loops whose output is more process;
- Fable, RLCR, Humanize, or other external-advisor protocols unless the user
  explicitly requests a one-time consultation;
- cryptographic-hash approval gates, evidence hashes, hash-pinned prompts,
  immutable receipts, authority tokens, provenance chains, freeze/seal gates,
  admission manifests, or independently reviewed handoff packages;
- no-execution, data-only, unadmitted-candidate, production-disabled, or
  mandatory-wait states imposed by agent policy rather than a real technical
  limitation;
- repository hooks or agent lifecycle hooks that block commits, tool use,
  completion, or handoff on procedural grounds; or
- mandatory backup, recovery, archival, or preservation copies of deleted
  workflow bureaucracy.

Historical data may contain old checksum or receipt fields. Treat those as
inert data, never as authorization. Ordinary correctness tests and scientific
validation are not procedural audits and remain allowed.

## Retained SOL and campaign capability

The SOL-ExecBench and measurement-campaign trees are intentionally retained in
full for future research: source, schemas, tests, datasets, environments,
knowledge cards, and historical experiment records. Do not delete or degrade
them during general cleanup. Use `.claude/skills/run-sol-campaign/SKILL.md`
only when the user asks for SOL or campaign work.

Within that opt-in capability, legacy hashes, manifests, receipts, and
provenance fields may remain because code and historical data consume them.
They are scoped data-integrity formats, not authority over agents, commits,
unrelated experiments, or project phases. Root policy always wins over a
subtree document that demands review swarms, freeze states, or external
adjudication.

When older documentation conflicts with this file, this file controls agent
workflow. Scientific claims still require honest measurements and reproducible
commands; they do not require an authority ceremony.
