# CUPTI PC-to-semantic-opcode surface projection (2026-09-30)

The native CUPTI campaign exposes PC offsets and aggregate reason vectors.  The
retained `control.json` sidecar maps those offsets to semantic opcode groups
(`memory`, `tensor`, `math`, `sfu`, `control`, `special`).
`tools/sm86_cupti_opcode_surface_projection.py` joins the two and fits a small
family-disjoint endpoint model.

This is deliberately not a physical route join: a sampled memory-group PC does
not identify LSU ownership, a tensor-group PC does not identify an issue slot,
and a branch state does not identify a scheduler path.  NCU values from the
held-out family are read only after prediction.

## Result

The semantic-PC join is negative and retained only as a falsifying committee
member. All 15 rows receive a candidate or explicit candidate-fallback
interval:

| phase | GPC MAPE | duration MAPE |
|---|---:|---:|
| cold | 47.92% | 51.02% |
| warm | 42.21% | 41.66% |

The result shows that more detailed PC-to-semantic grouping does not, by
itself, recover the hidden scheduler/cache state.  The lower-dimensional
aggregate CUPTI surface remains the useful non-promoted alternative.
