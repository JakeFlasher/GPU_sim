# Papers

This directory is a retained prior-art corpus for the non-root GPU
measurement and prediction project. It is not an active simulator roadmap.

The useful themes are:

- dynamic binary instrumentation and trace capture;
- CUPTI/Nsight Systems style runtime observation that does not require GPU
  hardware counter access;
- microbenchmarking results that explain visible timing and cache behavior;
- analytical and learned predictors for wall-clock time and cache behavior;
- simulator literature only as background for validation pitfalls, trace
  schemas, and metric definitions.

## Layout

- `knowledge_base/active/` — local PDFs retained as literature.
- `knowledge_base/inbox/` — unresolved PDFs kept for later manual triage.
- `knowledge_base/README.md` — how to use this corpus under the non-root
  measurement contract.
- `knowledge_base/problem_playbook.md` — quick routing from current research
  problems to the most relevant papers.
- `knowledge_base/selected_index.md` — compact index of high-signal papers for
  the current direction.
- `knowledge_base/manifest.json` and reports — historical paper metadata.
- `knowledge_base/archive/` — retained archived PDFs and related source
  material.
- `knowledge_base/inbox/` — retained unresolved PDFs for later manual triage.
- `notes/` — historical notes retained as source material, not as active
  roadmap direction.
- `scripts/` — historical paper-library maintenance scripts retained for
  provenance.
- `sources/` — historical paper source lists and overrides retained for
  provenance.

Historical simulator planning language in the guide files was cleaned up in
the May 3, 2026 pass, but paper source materials are preserved.
