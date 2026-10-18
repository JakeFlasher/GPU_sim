# SM86 constant-form L1 lookup residual ledger — 2026-10-16

H51 joins H48 form counters to exact independent NCU L1 lookup rows without
excluding observations or fitting parameters. Form groups are selected from
trace-side static counters only.

| form | hit joins | hit mean APE | miss joins | miss mean APE |
|---|---:|---:|---:|---:|
| other/plain | 32 | 5.85% | 43 | 31.53% |
| U16 | 42 | 51.36% | 50 | 34.90% |
| LTC128B | 0 positive-target | — | 3 | 26.68% |
| no constant form | 3 | 6.89% | 10 | 25.25% |

The LTC128B hit domain has zero positive NCU targets, so no hit APE is defined;
its global traffic remains modeled. U16 is materially different from
plain/other lookup behavior, but the residual is still broad and the form
probe found no stable timing differential.

Decision: retain source-qualified form observers, preserve global traffic, and
defer any U16/LTC lookup correction until a scope-matched reuse/working-set
probe establishes the client semantics. No NCU values enter replay or fitting.
