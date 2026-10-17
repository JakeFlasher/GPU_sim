# SM86 dirty-writeback scenario A/B — negative promotion — 2026-10-15

H47 exercised the existing bounded `CardedWriteback` path on five real
high-terminal-dirty traces. The capacity was predeclared at 4,096 sectors; it
was not selected from NCU values. Default full replay was compared with the
explicit writeback scenario at the same 825 GB/s service rate.

| workload | default GPC APE | writeback GPC APE | effect |
|---|---:|---:|---|
| qkv-rope standard | 29.19% | 24.92% | improves |
| RMSNorm forward | 12.34% | 11.51% | slight improve |
| Hyena FFT | 8.38% | 7.26% | slight improve |
| Flash attention noncausal | 23.72% | 23.72% | unchanged |
| fused linear/GELU/GRN | 24.69% | 26.16% | worsens |

SM-active frontiers are unchanged because writeback is a post-issue tail. The
mixed outcome does not identify a transferable capacity or service law, and
the existing probes do not observe hardware completion ownership. Writeback
timing remains an explicit opt-in scenario; default full replay is unchanged.
