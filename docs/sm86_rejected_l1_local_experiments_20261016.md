# Rejected SM86 L1/local experiments (full-corpus evidence)

All experiments listed here used complete 286-successful-root replay; none is
being presented as a partial-run result.

* H95d payload-only local expansion: 28.99% load / 14.02% store; rejected
  above the 10% bar.
* H97 U16-only resident-sector selection: L1 miss APE regressed
  18.71%→65.29%; rejected.
* H98 mixed-client LDGSTS partition: hit APE regressed 17.70%→18.22%; the
  0.09-point miss gain was smaller than the hit regression; rejected.

The accepted replacements are H99 LTC128B all-miss and H102 local
frame/coalescing correction.
* H103 partial-line pending-owner subtraction: full replay raised global L1
  hit/miss APE to 33.74%/62.44%; rejected and selector rolled back. The
  owner-kind fields remain observer-only for future evidence, never scored.
* H109/H110 subline observers did not change scored endpoints; H111's first
  DRAM GCC constant-sector promotion regressed DRAM read APE from 26.84% to
  203.56% and was rolled back. GCC traffic requires a correct per-GPC/L2
  deduplication ledger before promotion.
