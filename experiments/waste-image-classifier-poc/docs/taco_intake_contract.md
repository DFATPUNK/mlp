# TACO Intake Contract

Phase 0.8.2 creates a local review queue for TACO. It does not download TACO, approve TACO images for training, assemble V2 automatically, or make any claim about model quality.

## Review Queue, Not Direct Ingestion

The intake script parses a local COCO-style TACO annotation JSON and writes draft classifier and review-gate candidate CSVs. Every generated row has its training approval set to `false`. A human must review the local image, the category mapping, licence provenance, and routing suitability before copying or promoting a row into a V2 public-source candidate manifest.

## Full Images Only

This first intake uses original full images only. It does not create crops. A classifier draft row is generated only when a full image has exactly one annotation, the annotated object is prominent enough, the source category has an approved mapping to one of the six material labels, the local file exists, and the licence provenance is eligible for review.

The six classifier labels remain:

```txt
cardboard
glass
metal
paper
plastic
trash
```

`needs_review` is not a seventh classifier label.

## Classifier Versus Review Gate

The classifier dataset answers: "What material is this clear single primary item?"

The review-gate dataset answers: "Is this image suitable for automatic routing at all?"

Multi-object TACO scenes are not six-class classifier examples in this phase. They may become review-gate drafts with `auto_route_eligible=false` and `review_reason=multiple_objects`. Unsupported, ambiguous, or small-object cases can also help the future gate learn when automatic routing is unsafe.

## Object Area Threshold

For full-image classifier drafts, the single annotated object must meet the configured `--min-object-area-ratio`. The ratio is computed from the annotation bbox area divided by image width times height. A small mapped object is not treated as a clean material classifier example; it becomes a gate draft with `review_reason=ambiguous_scene`.

## Licence Status

`license_status` is a provenance filter, not a visual label.

- `eligible_for_review`: source URL, licence name, licence reference, and attribution are present, and the licence text does not visibly contain non-commercial or no-derivatives markers.
- `blocked`: licence metadata is missing, incomplete, unresolvable, non-commercial, or no-derivatives.
- `approved`: reserved for later human review, never emitted by the TACO intake script.

`eligible_for_review` is not training approval. It only means the row may be inspected locally. V2 assembly still requires `license_status=approved`, nonblank licence provenance fields, and explicit `approved_for_*_training=true` before any public-source row can enter a training manifest.
