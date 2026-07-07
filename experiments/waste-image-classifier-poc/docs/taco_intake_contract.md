# TACO Intake Contract

Phase 0.8.2 and 0.8.3 create a local review queue for TACO. The tooling does not download TACO images, approve TACO images for training, assemble V2 automatically, or make any claim about model quality.

## Review Queue, Not Direct Ingestion

The intake script parses a local COCO-style TACO annotation JSON. In `--plan-only` mode it writes category inventory, a licence evidence ledger, and a manual download plan without requiring local images. In normal mode it writes draft classifier and review-gate candidate CSVs for locally available images. Every generated candidate row has its training approval set to `false`. A human must review the local image, the category mapping, licence provenance, and routing suitability before copying or promoting a row into a V2 public-source candidate manifest.

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

For TACO only, a blank `images[].license` value may resolve to `CC BY 4.0` with reference `https://tacodataset.org/` under the named rule `taco_missing_license_default_cc_by_4_0`. This narrow rule is based on the official TACO Terms and Conditions statement that a missing licence entry defaults to CC BY 4.0. It applies only when the source dataset is TACO, the licence value is blank, an original source URL exists, and attribution text can be generated.

This resolution permits `eligible_for_review` only. It is intake provenance for human review, not final legal clearance and never training approval. Explicit `CC` remains blocked because it is insufficiently specific in the current metadata. `ODBL (c) OpenLitterMap & Contributors` remains blocked for automatic intake as well.

`eligible_for_review` is not training approval. It only means the row may be inspected locally. V2 assembly still requires `license_status=approved`, nonblank licence provenance fields, and explicit `approved_for_*_training=true` before any public-source row can enter a training manifest.
