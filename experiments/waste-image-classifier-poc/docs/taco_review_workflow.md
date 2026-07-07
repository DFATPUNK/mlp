# TACO Review Workflow

This workflow is manual by design. It creates a local review queue before any TACO-derived row can be promoted into V2.

1. Download TACO manually from the official project source and review the dataset licence and attribution requirements for the intended experiment.
2. Store annotations and images under ignored local paths, for example `data/public_sources/ingested/taco/`.
3. Copy `data/public_sources/taco_label_mapping.template.csv` to a local mapping CSV and add explicit source-category mapping rules. Do not use automatic label mapping.
4. Run a plan-only pass. This does not require local images and does not create classifier or gate candidate CSVs:

```bash
python scripts/prepare_taco_intake.py \
  --annotations data/public_sources/ingested/taco/annotations.json \
  --label-mapping data/public_sources/taco_label_mapping.csv \
  --output-dir data/public_sources/taco/review_outputs/plan_local \
  --min-object-area-ratio 0.20 \
  --plan-only
```

5. Inspect `taco_category_inventory.csv`, `taco_license_resolution.csv`, and `taco_download_plan.csv`. A blank TACO licence may be marked `eligible_for_review` as `CC BY 4.0` under the official TACO missing-licence default rule, but this is not training approval or legal clearance. Explicit `CC` and `ODBL` values remain blocked.
6. Manually download only the images you choose to review, using the local download plan as a queue.
7. Run the normal intake script against local image files:

```bash
python scripts/prepare_taco_intake.py \
  --annotations data/public_sources/ingested/taco/annotations.json \
  --image-root data/public_sources/ingested/taco/images \
  --label-mapping data/public_sources/taco_label_mapping.csv \
  --output-dir data/public_sources/taco/review_outputs/local_run \
  --min-object-area-ratio 0.20
```

8. Inspect `taco_review_gallery.html` locally. The gallery references local image files only and does not embed remote URLs.
9. Promote only selected rows manually by copying them into reviewed local public-source candidate CSVs and setting:

```txt
license_status=approved
approved_for_classifier_training=true
```

or:

```txt
license_status=approved
approved_for_gate_training=true
```

10. Run V2 assembly with the reviewed public candidate files.
11. Before future evaluation or model selection, create a fresh untouched external holdout. TACO-derived review data and promoted candidates are inspected training data, not a final benchmark.

This PR does not claim TACO, or any public dataset, is universally approved for commercial use.
