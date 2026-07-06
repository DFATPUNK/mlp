# Feedback Data Contract

Phase 0.7 adds an auditable feedback loop for the waste image classifier POC. The loop is deliberately offline and curated: predictions are reviewed by a human, stored as structured feedback, and only approved rows can later become candidate training data. This PR does not retrain a model.

## Classifier Labels Versus Gate Labels

Classifier labels remain exactly:

```txt
cardboard
glass
metal
paper
plastic
trash
```

`needs_review` is not a seventh TrashNet class. It is an operational workflow decision made when the calibrated confidence policy does not trust an automatic route.

`auto_route_eligible` is a separate human judgment. It means the image is sufficiently clear, single-object, in scope, and suitable for automatic routing. It is not derived from whether the current model was correct. A clear plastic bottle misclassified as glass can still be `auto_route_eligible=true`; a correctly predicted image with multiple objects should be `auto_route_eligible=false`.

## Confidence Is Not Unknown Detection

The current threshold policy uses calibrated top-class confidence to decide whether to auto-route or send an image to review. That threshold is useful, but it is not true unknown, unsupported-material, or out-of-distribution recognition. Low confidence can catch some uncertain cases, but high confidence can still occur on unsupported or ambiguous images. Human feedback records both the material label and whether automatic routing is appropriate.

## Curated Feedback, Not Automatic Retraining

Feedback rows are curated, versioned, reviewed, and approved before they become candidate datasets. This prevents accidental retraining on unresolved reviews, unsuitable images, mislabeled images, or diagnostic examples that would leak evaluation knowledge into model development.

Feedback must never trigger automatic retraining. A later training run should be an explicit experiment using a named manifest version, reviewed approvals, and a documented split strategy.

Two candidate datasets can be built later:

- A classifier candidate dataset, using approved confirmed or corrected material labels.
- A review-gate candidate dataset, using approved `auto_route_eligible` judgments and review reasons.

Neither candidate dataset is automatically used for model fitting in this phase.

## `external_diagnostic_v1` Lifecycle

The 20 inspected external images are named `external_diagnostic_v1`. They are useful diagnostics, not a blind final benchmark.

Lifecycle:

1. Initially diagnostic only: images are used only for qualitative error analysis.
2. Promoted feedback: individual images may be explicitly reviewed and copied into the feedback manifest.
3. Feedback-derived candidate data: approved feedback can be built into classifier or review-gate candidate manifests.

Once any `external_diagnostic_v1` row is promoted into feedback-derived candidate data, that external set cannot be reused as a comparative or final benchmark. A new untouched external final set is required for future model selection.
