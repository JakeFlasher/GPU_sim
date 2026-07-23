# Sources

This directory holds inputs and historical material used to build the paper knowledge base.

## Contents

- `FINAL_curated_paper_list.md`
- `arxiv_paper_list.md`
- `proceedings_paper_list.md`
  Original markdown paper lists. These are source catalogs, not the agent-facing working set.
- `manual_source_overrides.json`
  Override rules used by the build scripts when a paper needs a fixed local or remote source.

## Important Distinction

- `../knowledge_base/` is the canonical place Claude should search for paper-backed guidance.
- `sources/` exists to preserve provenance and rebuildability.

Do not add canonical paper copies here. New PDFs should go into `../knowledge_base/inbox/` and then be ingested.
