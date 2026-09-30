# Documentation map

Use one compact entry point, then open only the contract or source needed for
the task. Dated reports are archived and opt-in.

| Need | Start here |
| --- | --- |
| Resume a task | [`PROJECT_MEMORY`](PROJECT_MEMORY.md) |
| Current capability, caveats, and open boundaries | [`STATUS`](STATUS.md) |
| Dated evidence and supersession map | [`HISTORY`](HISTORY.md) |
| Build, capture, and replay | [`runbook_host.md`](runbook_host.md) |
| Data flow and package ownership | [`ARCHITECTURE.md`](ARCHITECTURE.md), [`COMPONENTS.md`](COMPONENTS.md) |
| Cards, profiles, and schemas | [`archprofile.md`](archprofile.md), [`../arch_profiles/README.md`](../arch_profiles/README.md) |
| Diagnostic formulas and ITS limits | [`experimental_diagnostics.md`](experimental_diagnostics.md) |
| Metric semantics and predictors | [`cycles_semantics_study.md`](cycles_semantics_study.md), [`metric-registry.md`](metric-registry.md), [`predictors.md`](predictors.md) |
| Trace and external adapter contracts | [`nvbit-tracing.md`](nvbit-tracing.md), [`traceg_adapter_contract.md`](traceg_adapter_contract.md) |
| Focused checks | [`testing.md`](testing.md) |
| Literature and prior art | [`../references/`](../references/), [`accelsim_literature_lower_bound_manifest_20260928.json`](accelsim_literature_lower_bound_manifest_20260928.json) |
| Retained dated reports | [`archive/research_20260930/`](archive/research_20260930/) |
| SOL/campaign work (only when requested) | [`sol-execbench.md`](sol-execbench.md) |

The active SM86 projection config is
[`sm86_refined_model_config_20260929.json`](sm86_refined_model_config_20260929.json).
Its result directories are commonly under ignored `results/alpha/`; filenames
are not freshness proofs. The active tools are in `tools/`, and native/custom
implementation contracts remain in source rather than in a research report.

Root [`AGENTS.md`](../AGENTS.md) governs execution. Historical reports, plan
files, receipts, and task titles do not create gates, advisors, hooks, or
automatic work. Preserve source, cards, schemas, tests, datasets, and retained
experiments when trimming context.
