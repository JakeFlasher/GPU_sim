# M5 Rule-1 provenance contract v2

The M5 controller accepts only an externally retained `M5_QUANTITY_AUTHORITY`
v2 artifact for new planning. The former embedded v1 quantity model remains
historical/prototype data and is quarantined from production planning.

The v2 authority is signed by the retained Ed25519 analysis authority and is
accepted only when all of the following are byte-bound:

- the v2 authority schema and its controller-pinned SHA-256;
- nonempty signed design-input digest bindings and complete source coverage for
  the execution order, scientific rank, and every quantity row;
- a controller-generated, retained, and revalidated mechanical ancestry audit
  over the exact design-input bytes, the variable scientific-authority projection,
  source registry and its closed contract, S001--S004 quarantine pins, normalized
  forbidden tokens, independent legacy dependency-edge and other structural
  fingerprints; JSON design inputs scan decoded values and all non-fixed keys,
  while schema-fixed policy keys are excluded explicitly; the audit policy
  byte-binds the exact self-contained tool bytes that the controller compiled,
  audit schema, registry schema, normalization/token rules, quarantine catalog,
  and structural-fingerprint catalogs;
- an ancestry declaration with no content-informing target, simulator-error,
  target-bracket, or derived-selection ancestors;
- at least one independent blind-review witness;
- an explicit operator-approval witness; and
- the out-of-band analysis-authority key used to verify the authority
signature.

M5 RUN, RESULT, and DONE use explicit v2 schema dispatch. v1 schemas remain
compatibility inputs and are never silently interpreted as v2. RUN records the
authority, declaration, audit, design-input, and review artifact references and
recomputes the audit from retained bytes before trusting the RUN. Result
revalidation derives family dependencies from each signed, source-covered quantity's
`upstream_family_ids`; no family name, quantity-group name, or special agreement
bound is supplied by the controller.

Readout independence is represented by a derived witness covering raw
ancestry, estimator form, mechanism, confounder sets, and falsification rule.
The authority also carries an exact consumer-binding registry and a versioned
gap mapping. Every typed gap retains disagreement statistics, a typed trigger,
and contradiction evidence when readouts disagree beyond the preregistered
limit.

The audit verdict is deliberately `NO_LISTED_REVIVAL_DETECTED`, not a semantic
ancestry or target-free approval. It proves declared-byte coverage and absence
of listed exact, normalized-token, and structural fingerprints only. Honest
completeness of declared influences, paraphrased or remembered contamination,
scientific validity, reviewer independence, and operator acceptance remain
external gates. The production trust registry remains empty.

The feature candidate is published and rehashed in an external immutable
publication before any NCU artifact is opened. NCU verdicts bind the candidate
model seal and remain additive cross-check evidence only. M5 collection remains
`M5_COLLECT_DISABLED`; this contract does not admit hardware evidence,
simulator inputs, calibration constants, architecture cards, or M7 credit.
