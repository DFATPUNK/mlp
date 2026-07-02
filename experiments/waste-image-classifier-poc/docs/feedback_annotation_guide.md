# Feedback Annotation Guide

Use the feedback manifest to separate material classification from automatic-routing suitability.

## 1. Correct Plastic Bottle

- Current prediction: `plastic`
- `human_outcome`: `confirmed`
- `confirmed_label`: `plastic`
- `auto_route_eligible`: `true`
- `review_reason`: `none`
- `approved_for_classifier_training`: `true`
- `approved_for_gate_training`: `true`

Use this when the image is a clear, single in-scope object and the model predicted the correct material.

## 2. Wrong Material, Clear Single Object

- Current prediction: `glass`
- `human_outcome`: `corrected`
- `confirmed_label`: `plastic`
- `auto_route_eligible`: `true`
- `review_reason`: `wrong_material`
- `approved_for_classifier_training`: `true`
- `approved_for_gate_training`: `true`

The image is still useful for classifier training because the human can provide a valid material label. It can also teach the gate that clear in-scope images may be eligible even when the current model was wrong.

## 3. Multiple Objects

- `human_outcome`: `rejected`
- `confirmed_label`: blank
- `auto_route_eligible`: `false`
- `review_reason`: `multiple_objects`
- `approved_for_classifier_training`: `false`
- `approved_for_gate_training`: `true`

Do not force a single TrashNet label when multiple primary objects are present.

## 4. Unsupported Object Or Non-Waste

- `human_outcome`: `rejected`
- `confirmed_label`: blank
- `auto_route_eligible`: `false`
- `review_reason`: `unsupported_material` or `non_waste`
- `approved_for_classifier_training`: `false`
- `approved_for_gate_training`: `true`

Unsupported or non-waste images are not new TrashNet classes. They are examples for the review gate.

## 5. Blurry Or Too Dark

- `human_outcome`: `rejected`
- `confirmed_label`: blank
- `auto_route_eligible`: `false`
- `review_reason`: `low_quality`
- `approved_for_classifier_training`: `false`
- `approved_for_gate_training`: `true`

Image quality failures should teach the system to ask for review, not to guess a material class.

## 6. Ambiguous Scene With A Material Suggestion

- Current prediction: any six-class material suggestion
- `human_outcome`: `rejected`
- `confirmed_label`: blank
- `auto_route_eligible`: `false`
- `review_reason`: `ambiguous_scene`
- `approved_for_classifier_training`: `false`
- `approved_for_gate_training`: `true`

The model may suggest a plausible material, but the scene is unsafe for automatic routing.
