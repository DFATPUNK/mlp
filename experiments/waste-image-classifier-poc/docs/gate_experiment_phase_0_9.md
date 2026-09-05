# Phase 0.9 CLIP binary gate experiment

Phase 0.9 is a local experiment, not production routing or deployment code. It asks whether frozen CLIP ViT-B/32 image embeddings plus a balanced scikit-learn `LogisticRegression` can recognize images that should be sent to human review before the existing six-class material policy is allowed to auto-route.

The gate target is separate from the material label:

```txt
auto_route_eligible=true|false
```

`needs_review` remains an operational decision, not a seventh material class. The combined policy is conservative:

```txt
gate=false -> needs_review
gate=true  -> preserve the existing V1 classifier route
```

## Baseline and interpretation

On the 37 reviewed TACO gate-challenge images, V1 sent 17 to review and unsafely auto-routed 20. Its expected-review recall and policy-safe outcome rate were both `0.4594594594594595`.

The immediate interpretation target is to reduce unsafe auto-routes from 20/37 to ideally no more than 5/37 without simply sending every image to review. This target is diagnostic, not a test assertion or a performance claim.

## Mode A: honest mini-holdout

Mode A trains with clean TrashNet training images as positive examples and reviewed feedback gate rows as negative examples. The 37 reviewed TACO images are used only for evaluation.

This is the useful first generalization check because the TACO challenge images do not enter gate fitting. It is still small and previously inspected, so it is not a final benchmark.

```bash
PY="/Users/jeremy/.pyenv/versions/3.10.13/bin/python"
V2_DIR="data/v2/conservative_feedback_taco_gate_20260708_182336"
TACO_EVAL="data/external_evals/taco_gate_reviewed_v1_baseline/external_manifest.csv"
V1_EXTERNAL_OUT="artifacts/external_evals/taco_gate_reviewed_v1_baseline_on_poc_clip_vit_b32"

"$PY" scripts/build_gate_manifest.py \
  --v2-classification-manifest "$V2_DIR/v2_classification_manifest.csv" \
  --v2-gate-manifest "$V2_DIR/v2_gate_manifest.csv" \
  --negative-source-filter feedback \
  --trashnet-image-root data/raw/trashnet-source \
  --feedback-image-root data/external_images/images \
  --public-image-root data/public_sources/ingested/taco-selected \
  --output-dir artifacts/gate_experiments/phase_0_9_feedback_only_manifest

"$PY" scripts/train_clip_logreg_gate.py \
  --gate-manifest artifacts/gate_experiments/phase_0_9_feedback_only_manifest/gate_manifest.csv \
  --output-dir artifacts/gate_experiments/phase_0_9_feedback_only_gate \
  --max-positive-examples 300 \
  --seed 42

"$PY" scripts/evaluate_clip_logreg_gate.py \
  --gate-model artifacts/gate_experiments/phase_0_9_feedback_only_gate/gate_model.joblib \
  --external-manifest "$TACO_EVAL" \
  --image-root data/external_evals/taco_gate_reviewed_v1_baseline/images \
  --expected-auto-route-eligible false \
  --output-dir artifacts/gate_experiments/phase_0_9_feedback_only_on_taco_reviewed

"$PY" scripts/evaluate_combined_policy.py \
  --v1-external-predictions "$V1_EXTERNAL_OUT/external_predictions.csv" \
  --gate-predictions artifacts/gate_experiments/phase_0_9_feedback_only_on_taco_reviewed/gate_predictions.csv \
  --output-dir artifacts/gate_experiments/phase_0_9_feedback_only_combined_policy
```

## Mode B: in-sample sanity check

Mode B adds reviewed public/TACO gate rows to the negative training examples and then evaluates on those same reviewed TACO images. Its output must be labelled `in_sample_sanity_check_not_final_benchmark`.

This mode is only an upper-bound sanity check for whether the signal is learnable. It is not evidence of generalization and must not be presented as a final benchmark.

```bash
"$PY" scripts/build_gate_manifest.py \
  --v2-classification-manifest "$V2_DIR/v2_classification_manifest.csv" \
  --v2-gate-manifest "$V2_DIR/v2_gate_manifest.csv" \
  --negative-source-filter all \
  --trashnet-image-root data/raw/trashnet-source \
  --feedback-image-root data/external_images/images \
  --public-image-root data/public_sources/ingested/taco-selected \
  --output-dir artifacts/gate_experiments/phase_0_9_feedback_plus_taco_manifest

"$PY" scripts/train_clip_logreg_gate.py \
  --gate-manifest artifacts/gate_experiments/phase_0_9_feedback_plus_taco_manifest/gate_manifest.csv \
  --output-dir artifacts/gate_experiments/phase_0_9_feedback_plus_taco_gate \
  --max-positive-examples 300 \
  --seed 42

"$PY" scripts/evaluate_clip_logreg_gate.py \
  --gate-model artifacts/gate_experiments/phase_0_9_feedback_plus_taco_gate/gate_model.joblib \
  --external-manifest "$TACO_EVAL" \
  --image-root data/external_evals/taco_gate_reviewed_v1_baseline/images \
  --expected-auto-route-eligible false \
  --evaluation-label in_sample_sanity_check_not_final_benchmark \
  --output-dir artifacts/gate_experiments/phase_0_9_feedback_plus_taco_on_taco_reviewed

"$PY" scripts/evaluate_combined_policy.py \
  --v1-external-predictions "$V1_EXTERNAL_OUT/external_predictions.csv" \
  --gate-predictions artifacts/gate_experiments/phase_0_9_feedback_plus_taco_on_taco_reviewed/gate_predictions.csv \
  --output-dir artifacts/gate_experiments/phase_0_9_feedback_plus_taco_combined_policy
```

## Artifacts and next evidence

The training script caches normalized frozen CLIP embeddings, limits positive examples deterministically when requested, and fits only a balanced logistic-regression classifier. The gate model, embedding cache, manifests used, predictions, and reports remain under ignored local artifact paths.

Final proof requires a fresh external holdout of 40–60 never-used images. Success on that holdout means substantially fewer unsafe auto-routes while retaining useful automatic-routing coverage; a gate that sends everything to review is not a useful success.
