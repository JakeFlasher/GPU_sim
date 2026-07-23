# M5 Rule-1 provenance contract v2

The M5 controller accepts only an externally retained `M5_QUANTITY_AUTHORITY`
v2 artifact for new planning. The former embedded v1 quantity model remains
historical/prototype data and is quarantined from production planning.

The v2 authority is signed by the retained Ed25519 analysis authority and is
accepted only when all of the following are byte-bound:

- the v2 authority schema and its controller-pinned SHA-256;
- an ancestry witness with no content-informing target, simulator-error,
target-bracket, or derived-selection ancestors;
- at least one independent blind-review witness;
- an explicit operator-approval witness; and
- the out-of-band analysis-authority key used to verify the authority
signature.

M5 RUN, RESULT, and DONE use explicit v2 schema dispatch. v1 schemas remain
compatibility inputs and are never silently interpreted as v2. RUN records the
authority, ancestry, and review artifact references and asserts target-free
operator approval. Result revalidation derives family dependencies from RUN,
including `b -> d` and `b,d -> f`, rather than reconstructing a separate graph.

Readout independence is represented by a derived witness covering raw
ancestry, estimator form, mechanism, confounder sets, and falsification rule.
The authority also carries an exact consumer-binding registry and a versioned
gap mapping. Every typed gap retains disagreement statistics, a typed trigger,
and contradiction evidence when readouts disagree beyond the preregistered
limit.

The feature candidate is published and rehashed in an external immutable
publication before any NCU artifact is opened. NCU verdicts bind the candidate
model seal and remain additive cross-check evidence only. M5 collection remains
`M5_COLLECT_DISABLED`; this contract does not admit hardware evidence,
simulator inputs, calibration constants, architecture cards, or M7 credit.
