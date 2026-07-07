# Public Dataset Intake

Phase 0.8 does not import a public dataset. It creates the contract for doing that later without silent label mapping or evaluation leakage.

## Intake Process

1. Acquire a candidate public dataset manually.
2. Verify the licence and usage constraints for the intended experiment.
3. Store local data under ignored paths such as `data/public_sources/ingested/`.
4. Inspect the dataset's source labels and annotation format.
5. Create explicit mapping rules in `data/public_sources/label_mapping.csv`.
6. Create classifier and/or review-gate candidate CSVs from reviewed source rows.
7. Run `scripts/build_v2_dataset.py` to assemble V2 manifests.
8. Inspect `v2_dataset_report.json` before any training experiment uses the outputs.

Do not assume a public label means the same thing as a TrashNet material class. Public labels can be object names, scene tags, material tags, or dataset-specific categories. Only approved mapping rules may generate classifier rows.

Public-source candidate rows also preserve licence and attribution evidence: `source_url`, `source_license`, `source_license_reference`, `source_attribution`, `license_status`, `source_annotation_id`, and `object_area_ratio`. V2 assembly accepts public training rows only after human review sets `license_status=approved` and the relevant `approved_for_*_training` field to `true`.

For TACO-specific local intake, see `taco_intake_contract.md` and `taco_review_workflow.md`. The TACO script produces review drafts only; it never marks rows as approved.

## Local Data

Real public-source images, downloaded metadata, converted candidate CSVs, and generated V2 manifests are local experiment data. They are intentionally ignored by Git. Track templates and documentation only.

This PR does not claim that any specific public dataset is approved for commercial use.
