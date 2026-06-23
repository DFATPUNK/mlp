# Waste Image Classifier POC

This isolated experiment tests whether a pretrained EfficientNet-B0 can classify single-primary-item waste photos into the six TrashNet classes and whether calibrated confidence can support an honest `needs_review` routing policy.

## 1. What this POC proves

- Trains a real TorchVision EfficientNet-B0 transfer-learning model on TrashNet.
- Evaluates validation and untouched test performance from a deterministic manifest.
- Fits temperature scaling on validation logits only.
- Selects a `needs_review` threshold from validation data only.
- Supports a manually curated 20-image qualitative external challenge set.

## 2. What it does not prove

- It is not object detection.
- It does not identify multiple distinct objects in one image.
- It does not prove industrial waste-sorting readiness.
- It does not make `needs_review` a seventh class; `trash` remains a real TrashNet class.

## 3. Directory structure

```txt
configs/                         EfficientNet-B0 baseline YAML
scripts/                         Download, manifest, train, evaluate, predict, orchestrate
src/waste_poc/                   Reusable experiment code
tests/                           Focused unit tests
data/raw/                        Gitignored TrashNet clone and metadata
data/manifests/                  Gitignored generated manifests
data/external_images/            Committed manifest template; images are gitignored
artifacts/                       Gitignored runs, models, plots, reports, model cards
docs/                            External-image checklist and model-card template
```

## 4. Dataset provenance and class contract

The downloader uses only the original upstream repository: `https://github.com/garythung/trashnet.git`. It records the resolved Git commit in `data/raw/trashnet_source_metadata.json` and searches for folders containing all six expected classes: cardboard, glass, metal, paper, plastic, and trash.

`needs_review` is never accepted as a training label. It is a routing policy outcome when calibrated confidence is below the selected validation threshold.

## 5. Google Colab quick start

Use a GPU runtime when available. Colab usually includes PyTorch and TorchVision, so do not blindly reinstall Torch or TorchVision.

```bash
!git clone https://github.com/DFATPUNK/mlp.git
%cd mlp
!pip install numpy pandas Pillow PyYAML scikit-learn matplotlib seaborn tqdm
%cd experiments/waste-image-classifier-poc
!python scripts/run_poc.py --config configs/efficientnet_b0_baseline.yaml
```

If Colab does not include a compatible Torch/TorchVision pair, install one using the official PyTorch instructions for the selected runtime.

## 6. Local setup

Install a compatible Torch and TorchVision pair through the official PyTorch installation instructions for your CPU/GPU platform. Then install the experiment dependencies:

```bash
cd experiments/waste-image-classifier-poc
python -m pip install -r requirements.txt
```

Training can run on CPU but will be slower. The code prints the selected device and uses mixed precision only when CUDA is available.

## 7. Reproducible commands

```bash
python scripts/download_trashnet.py
python scripts/build_manifest.py --config configs/efficientnet_b0_baseline.yaml
python scripts/train.py --config configs/efficientnet_b0_baseline.yaml --mode frozen_backbone
python scripts/evaluate.py --checkpoint artifacts/runs/<run>/best_model.pt
```

One-command orchestration:

```bash
python scripts/run_poc.py --config configs/efficientnet_b0_baseline.yaml
```

Optional flags: `--skip-download`, `--skip-manifest`, `--skip-fine-tune`, `--skip-external`, and `--run-name <name>`.

## 8. Frozen-backbone baseline

The frozen baseline freezes the pretrained EfficientNet-B0 backbone and trains only the replacement six-class classifier head. Checkpoints are selected by validation macro F1.

## 9. Fine-tuning run

Fine-tuning starts from the best frozen-backbone checkpoint, unfreezes only the final EfficientNet feature stage plus the classifier head, and uses a smaller learning rate.

## 10. Calibration and threshold selection

`evaluate.py` fits temperature scaling using validation logits only and writes `temperature_scaling.json`. The threshold policy accepts predictions only when calibrated top-class confidence reaches the selected validation threshold; otherwise it routes to `needs_review`.

The test split is evaluated only after calibration and threshold selection are frozen.

## 11. Adding the 20 external images

Add images manually to `data/external_images/images/` and fill `data/external_images/external_manifest.csv`. Follow `docs/EXTERNAL_IMAGES_CHECKLIST.md`. The scripts do not download external images from Google Images, random websites, or copyright-unclear sources.

## 12. How to read the reports

A completed run writes metrics, confusion matrices, calibration plots, threshold policy, test-policy report, external challenge outputs, and a generated `model_card.md` under `artifacts/runs/<run>/`.

## 13. Criteria for moving to an MLP image-pipe implementation

Consider integration only if validation/test performance is credible, calibration is reasonable, the `needs_review` policy achieves the target auto-route precision at useful coverage, and the external challenge set surfaces acceptable failure modes.

## 14. Limitations

TrashNet is controlled and small compared with real workflow images. The external 20-image set is qualitative, not statistically representative. Confidence is a routing signal, not a factual guarantee. Low-confidence, ambiguous, multi-item, low-light, and blurred inputs should go to human review.

## Phase 0.5 corrective external evaluation

This phase fixes EXIF orientation handling, external policy metrics, device selection, macOS DataLoader defaults, and report overwrite protection without retraining. NumPy is pinned to `numpy>=1.26,<2` because the supported local Torch environment for this POC can fail with NumPy 2. PyTorch and TorchVision should still be installed separately according to the user's CPU, CUDA, MPS, or Colab runtime; this experiment does not pin a CPU-only Torch wheel.

External row output fields are intentionally explicit:

- `known_label_top1_correct`: top-1 prediction correctness for rows with an expected label; blank when no expected label exists.
- `is_auto_routed`: the calibrated policy routed directly to a material class.
- `is_needs_review`: the calibrated policy abstained for human review.
- `auto_route_is_correct`: an auto-routed known-label row was routed to the expected class; blank without an expected label.
- `is_safe_abstention`: a known-label row was conservatively sent to `needs_review`.
- `expected_review_correctly_routed`: a row expected to need review was sent to `needs_review`.
- `is_unsafe_auto_route`: the policy auto-routed to the wrong known class or auto-routed an expected-review image.
- `policy_outcome_safe`: correct auto-route or conservative review outcome for the row contract.

Preserve the old external evaluation output manually if it exists:

```bash
RUN="artifacts/runs/poc_frozen_backbone_fine_tune"

mv "$RUN/external_evaluation" \
  "$RUN/external_evaluation_before_phase_0_5"
```

Then rerun the same checkpoint and same 20 external images without retraining:

```bash
python scripts/evaluate_external.py \
  --checkpoint "$RUN/best_model.pt" \
  --temperature "$RUN/temperature_scaling.json" \
  --threshold-policy "$RUN/threshold_policy.json" \
  --external-manifest data/external_images/external_manifest.csv \
  --output-dir "$RUN/external_evaluation_exif_fixed" \
  --device auto
```

Compare `external_summary.json`, `external_failure_notes.md`, and the two gallery PNGs between the old and new folders. The important comparisons are whether iPhone images now display upright, whether unsafe automatic routes are counted separately from safe abstentions, and whether the selected device appears once per run.
