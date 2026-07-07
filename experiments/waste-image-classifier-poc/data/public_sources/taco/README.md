# Local TACO Intake Data

This directory documents the local TACO review workflow. Do not commit real TACO annotations, images, generated draft candidates, category inventories, galleries, or promoted candidate CSVs.

Suggested ignored local layout:

```txt
data/public_sources/ingested/taco/
data/public_sources/taco/review_outputs/
data/public_sources/taco/taco_classifier_candidates.csv
data/public_sources/taco/taco_gate_candidates.csv
data/public_sources/taco/taco_category_inventory.csv
```

Run `scripts/prepare_taco_intake.py` against a manually downloaded local TACO COCO-style JSON and local image root. The script writes draft rows with training approval set to `false`; a human must review category mappings, licence provenance, and visual suitability before any row is copied into a V2 candidate manifest.
