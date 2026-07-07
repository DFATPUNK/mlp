# Public Source Candidate Data

This directory holds templates for future public-dataset intake. Real public-source files are local data and must not be committed.

Ignored local paths:

- `data/public_sources/ingested/`
- `data/public_sources/ingested/taco/`
- `data/public_sources/public_classifier_candidates.csv`
- `data/public_sources/public_gate_candidates.csv`
- `data/public_sources/label_mapping.csv`
- `data/public_sources/taco/review_outputs/`
- `data/public_sources/taco/taco_classifier_candidates.csv`
- `data/public_sources/taco/taco_gate_candidates.csv`
- `data/public_sources/taco/taco_category_inventory.csv`
- `data/v2/`

Use the template CSVs as starting points, then create reviewed local manifests outside Git. Phase 0.8 does not download or approve any public dataset. Phase 0.8.2 adds a TACO-specific local review queue; see `taco/README.md` and `../../docs/taco_review_workflow.md`.
