# Model Card Template: Single-item waste image routing demonstration

## Purpose
Single-item waste image routing demonstration.

## Intended use
Whole-image classification of one primary waste item into the six TrashNet material classes.

## Non-intended use
This is not object detection, does not identify multiple items individually, and must not be used as the sole basis for industrial waste-sorting decisions.

## Source dataset
- Repository: https://github.com/garythung/trashnet.git
- Resolved commit SHA: filled by run artifacts

## Classes
cardboard, glass, metal, paper, plastic, trash

## Split methodology
Deterministic stratified 70/15/15 manifest split with exact duplicate SHA256 groups kept in one split.

## Model architecture
EfficientNet-B0 transfer learning.

## Calibration and routing policy
`needs_review` is an inference routing outcome when calibrated confidence is below the selected validation threshold. It is not a training class.

## Known limitations
Controlled TrashNet images may not represent workflow images. Ambiguous, multi-item, low-light, or blurred images should route to human review.
